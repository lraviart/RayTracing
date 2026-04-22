import numpy as np
from scipy import optimize

c_0 = 299792458  # Speed of light in m/s

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


class Rx:

    def __init__(self, name, position):

        self.name = name
        self.position = np.array(position)


class Tdoa:

    def __init__(self, Rx1, Rx2, delay1, delay2):

        self.Rx1 = Rx1
        self.Rx2 = Rx2
        self.tdoa = delay1 - delay2


    
class TdoaLocalization:

    def __init__(self, first_toas, RX_positions):

        self.RXs = []
        for i in range(len(RX_positions)):
            self.RXs.append(Rx(f"Rx{i}", RX_positions[i]))

        self.tdoas = []

        # Find reference delay
        self.ref_index = -1
        for i in range(len(first_toas)):
            if first_toas[i] > 0:
                self.ref_index = i
                break
        if self.ref_index == -1:
            raise ValueError("No valid delay found")
        
        # Calculate TDOAs
        for i in range(self.ref_index+1, len(first_toas)):
            if first_toas[i] > 0:
                # print(i)
                self.tdoas.append(Tdoa(self.RXs[i], self.RXs[self.ref_index], first_toas[i], first_toas[self.ref_index]))
        


    def initial_guess(self):

        # Initial guess: mean of Rx positions with valid delays
        valid_RXs = []
        for tdoa in self.tdoas:
            if tdoa.Rx1 not in valid_RXs:
                valid_RXs.append(tdoa.Rx1)
            if tdoa.Rx2 not in valid_RXs:
                valid_RXs.append(tdoa.Rx2)
    
        if not valid_RXs:
            raise ValueError("No valid RX positions found")

        # Calculate the mean position
        guess = np.mean([rx.position for rx in valid_RXs], axis=0)

        return guess


    def mse_function(self, guess):
        cost = 0
        for tdoa in self.tdoas:
            distance = dist(guess, tdoa.Rx1.position) - dist(guess, tdoa.Rx2.position)
            cost += (tdoa.tdoa * c_0 - distance)**2
        return cost
    

    def localize(self):
        point = optimize.minimize(fun=self.mse_function, x0=self.initial_guess(), args=(), method='Nelder-Mead')

        return point.x
    

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
            solution = u + root1 * v + self.RXs[self.ref_index].position
        elif root2 > 0 and root1 < 0:
            solution = u + root2 * v + self.RXs[self.ref_index].position
        elif root1 > 0 and root2 > 0:
            solution1 = u + root1 * v + self.RXs[self.ref_index].position
            solution2 = u + root2 * v + self.RXs[self.ref_index].position
            # Choose the solution that minimizes the MSE
            if self.mse_function(solution1) < self.mse_function(solution2):
                solution = solution1
            else:
                solution = solution2
        else:
            raise ValueError("No valid solution found")
        
        return solution