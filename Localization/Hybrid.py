import numpy as np
from scipy import optimize
from Tdoa import Rx, Tdoa
from Aoa import Aoa

c_0 = 299792458  # Speed of light in m/s

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


class HybridLocalization:

    

    def __init__(self, RX_positions, first_toas, first_aoas, amplitudes, second_toas=None, second_aoas=None, second_amplitudes=None, ref_index=None):

        self.Rxs = []
        for i in range(len(RX_positions)):
            self.Rxs.append(Rx(f"Rx{i}", RX_positions[i]))
        self.valid = [True if first_toas[i] > 0 else False for i in range(len(first_toas))]
        self.first_toas = first_toas
        self.first_amplitudes = amplitudes

        # Find reference delay
        if ref_index is not None:
            if not self.valid[ref_index]:
                raise ValueError("Reference index must have a valid delay")
            self.ref_index = ref_index
        else:
            self.ref_index = -1
            for i in range(len(first_toas)):
                if self.valid[i]:
                    self.ref_index = i
                    break
            if self.ref_index == -1:
                raise ValueError("No valid delay found")
        
        # Calculate TDOAs
        self.get_tdoas(self.ref_index)

        self.first_aoas = []
        for i in range(len(first_aoas)):
            if first_aoas[i] < 2*np.pi:
                self.first_aoas.append(Aoa(self.Rxs[i], first_aoas[i]))

        self.second_toas = second_toas
        self.second_aoas = second_aoas
        self.second_amplitudes = second_amplitudes
        

    def get_tdoas(self, ref_index):

        if not self.valid[ref_index]:
            raise ValueError("Reference index must have a valid delay")

        self.ref_index = ref_index
        
        tdoas = []
        for i in range(len(self.first_toas)):
            if i != ref_index and self.valid[i]:
                tdoas.append(Tdoa(self.Rxs[i], self.Rxs[ref_index],
                                  self.first_toas[i], self.first_toas[ref_index],  
                                  self.first_amplitudes[i], self.first_amplitudes[ref_index]))
        self.tdoas = tdoas

        return tdoas

    
    def initial_guess(self):

        # Initial guess: mean of Rx positions with valid angles and delays
        valid_Rxs = []
        for tdoa in self.tdoas:
            if tdoa.Rx1 not in valid_Rxs:
                valid_Rxs.append(tdoa.Rx1)
            if tdoa.Rx2 not in valid_Rxs:
                valid_Rxs.append(tdoa.Rx2)

        for aoa in self.first_aoas:
            if aoa.RX not in valid_Rxs:
                valid_Rxs.append(aoa.RX)

        # Calculate the mean position
        guess = np.mean([rx.position for rx in valid_Rxs], axis=0)

        return guess
    

    def angle_sector(self, angle):
        # 1: 0 to 90, 2: 90 to 180, 3: -180 to -90, 4: -90 to 0
        return (int(angle // (np.pi/2)) + 4) % 4 + 1

    
    def identify_wall(self, first_angle, second_angle):

        first_sector = self.angle_sector(first_angle)
        second_sector = self.angle_sector(second_angle)

        if first_sector == second_sector:
            if first_sector in [1, 3]:
                if second_angle < first_angle:
                    return 'vertical'
                else:
                    return 'horizontal'
            else:
                if second_angle < first_angle:
                    return 'horizontal'
                else:
                    return 'vertical'
                
        elif first_sector == 1:
            if second_sector == 2:
                return 'vertical'
            elif second_sector == 4:
                return 'horizontal'
        elif first_sector == 2:
            if second_sector == 1:
                return 'vertical'
            elif second_sector == 3:
                return 'horizontal'
        elif first_sector == 3:
            if second_sector == 2:
                return 'horizontal'
            elif second_sector == 4:
                return 'vertical'
        elif first_sector == 4:
            if second_sector == 1:
                return 'vertical'
            elif second_sector == 3:
                return 'horizontal'

        return None  # No wall identified
    

    def mse(self, estimate):

        cost = 0
        for tdoa in self.tdoas:
            distance = dist(estimate, tdoa.Rx1.position) - dist(estimate, tdoa.Rx2.position)
            cost += (tdoa.tdoa * 3e8 - distance)**2

        for aoa in self.first_aoas:
            cost += aoa.cost(estimate)**2

        return cost
    

    def mse2(self, estimate):

        cost = 0
        for i in range(len(self.Rxs)):
            if self.valid[i]:
                distance = dist(estimate, self.Rxs[i].position) - dist(estimate, self.Rxs[(i+1) % len(self.Rxs)].position)
                cost += ((self.first_toas[i] - self.first_toas[(i+1) % len(self.first_toas)]) * c_0 - distance)**2
        for aoa in self.first_aoas:
            cost += aoa.cost(estimate)**2
        return cost


    def mse_time(self, estimate):

        x, y, t0 = estimate
        cost = 0
        for i in range(len(self.Rxs)):
            if self.valid[i]:
                toa = self.first_toas[i] - t0
                point = self.Rxs[i].position + \
                        toa * c_0 * np.array([np.cos(self.first_aoas[i].angle), 
                                              np.sin(self.first_aoas[i].angle)])
                cost += np.linalg.norm(point - np.array([x, y]))**2

        return cost

    
    def mse_time2(self, estimate):

        x, y, t0 = estimate
        cost = 0
        for i in range(len(self.Rxs)):
            if self.valid[i]:
                toa = self.first_toas[i] - t0
                cost += (toa * c_0 - dist(estimate[:2], self.Rxs[i].position))**2
                cost += self.first_aoas[i].cost(estimate[:2])**2
        return cost
    


    def mse_second(self, prior, estimate):

        cost = 0
        t0 = np.zeros(len(self.Rxs))
        alpha0 = np.zeros(len(self.Rxs))
        for i in range(len(self.Rxs)):
            t0[i] = self.first_toas[i] - dist(prior, self.Rxs[i].position) / c_0
            alpha0[i] = np.arctan2(prior[1] - self.Rxs[i].position[1], prior[0] - self.Rxs[i].position[0])

        for i in range(len(self.Rxs)):
            if self.valid[i]:
                for j in range(len(self.second_toas[i])):
                    delta_t = self.second_toas[i][j] - t0[i]
                    angle = self.second_aoas[i][j]
                    wall = self.identify_wall(alpha0[i], angle)
                    # print(f"Wall identified for Rx{i}: {wall}")
                    if wall == 'vertical':
                        cost += (self.Rxs[i].position[1] + delta_t * c_0 * np.sin(angle) - estimate[1])**2
                    elif wall == 'horizontal':
                        cost += (self.Rxs[i].position[0] + delta_t * c_0 * np.cos(angle) - estimate[0])**2

        return cost
    
    

    def localize(self):

        point = optimize.minimize(self.mse, self.initial_guess(), method='Nelder-Mead')

        return point.x
    
    def localize2(self):

        point = optimize.minimize(self.mse2, self.initial_guess(), method='Nelder-Mead')

        return point.x
    
    def localize_time(self):

        point = optimize.minimize(self.mse_time, np.append(self.initial_guess(), -2e-8), method='Nelder-Mead')

        return point.x[:2], point.x[2]
    
    
    def localize_with_second(self):

        prior = self.localize_least_squares()

        point = optimize.minimize(lambda x:  self.mse_second(prior, x), prior, method='Nelder-Mead')

        return point.x
    
    def localize_iterative(self, max_iterations=10, tolerance=1e-6):

        estimate = self.localize_least_squares()

        for _ in range(max_iterations):
            prior = estimate
            estimate = optimize.minimize(lambda x: self.mse(x) + self.mse_second(prior, x), prior, method='Nelder-Mead').x
            if np.linalg.norm(estimate - prior) < tolerance:
                break

        return estimate
    

    def localize_least_squares(self):

        A = np.zeros((len(self.tdoas), 2))
        b = np.zeros(len(self.tdoas))
        c = np.zeros(len(self.tdoas))

        for i, tdoa in enumerate(self.tdoas):
            A[i, 0] = tdoa.Rx1.position[0] - tdoa.Rx2.position[0]
            A[i, 1] = tdoa.Rx1.position[1] - tdoa.Rx2.position[1]
            b[i] = 1/2 * ((tdoa.Rx1.position[0] - tdoa.Rx2.position[0])**2 + \
                          (tdoa.Rx1.position[1] - tdoa.Rx2.position[1])**2 - \
                          (tdoa.tdoa * c_0)**2)
            c[i] = -(tdoa.tdoa) * c_0

        # Solve the least squares problem
        A_proj = np.linalg.inv(A.T @ A) @ A.T
        u = A_proj @ b
        v = A_proj @ c


        # Solve the quadratic equation
        quad_a = 1 - v.T @ v
        quad_b = -2 * u.T @ v
        quad_c = -u.T @ u
        discriminant = quad_b**2 - 4*quad_a*quad_c
        if discriminant < 0:
            raise ValueError("No solution found")
        
        root1 = (-quad_b + np.sqrt(discriminant)) / (2*quad_a)
        root2 = (-quad_b - np.sqrt(discriminant)) / (2*quad_a)

        # Choose the solution
        if root1 > 0 and root2 < 0:
            solution = u + root1 * v + self.Rxs[self.ref_index].position
        elif root2 > 0 and root1 < 0:
            solution = u + root2 * v + self.Rxs[self.ref_index].position
        elif root1 > 0 and root2 > 0:
            solution1 = u + root1 * v + self.Rxs[self.ref_index].position
            solution2 = u + root2 * v + self.Rxs[self.ref_index].position
            # Choose the solution that minimizes the MSE
            if self.mse_function(solution1) < self.mse_function(solution2):
                solution = solution1
            else:
                solution = solution2
        else:
            raise ValueError("No valid solution found")
        
        return solution