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

    def __init__(self, Rx1, Rx2, delay1, delay2, a1, a2):

        self.Rx1 = Rx1
        self.Rx2 = Rx2
        self.tdoa = delay1 - delay2
        self.a1 = a1
        self.a2 = a2


    
class TdoaLocalization:

    def __init__(self, RX_positions, first_toas, amplitudes):

        self.RXs = []
        for i in range(len(RX_positions)):
            self.RXs.append(Rx(f"Rx{i}", RX_positions[i]))
        self.valid = [True if first_toas[i] > 0 else False for i in range(len(first_toas))]
        self.first_toas = first_toas
        self.amplitudes = amplitudes

        # Find reference delay
        self.ref_index = -1
        for i in range(len(first_toas)):
            if self.valid[i]:
                self.ref_index = i
                break
        if self.ref_index == -1:
            raise ValueError("No valid delay found")
        
        # Calculate TDOAs
        self.get_tdoas(self.ref_index)
        

    def get_tdoas(self, ref_index):

        if not self.valid[ref_index]:
            raise ValueError("Reference index must have a valid delay")

        self.ref_index = ref_index
        
        tdoas = []
        for i in range(len(self.first_toas)):
            if i != ref_index and self.valid[i]:
                tdoas.append(Tdoa(self.RXs[i], self.RXs[ref_index],
                                  self.first_toas[i], self.first_toas[ref_index],  
                                  self.amplitudes[i], self.amplitudes[ref_index]))
        self.tdoas = tdoas

        return tdoas


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
    

    def localize_weighted_least_squares(self, iter=10):

        W = np.eye(len(self.tdoas))
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

        residuals_record = np.zeros((iter, len(self.tdoas)))
        for i in range(iter):
            
            # Solve the least squares problem
            A_proj = np.linalg.inv(A.T @ W @ A) @ A.T @ W
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
                root = root1
                solution = u + root1 * v + self.RXs[self.ref_index].position
            elif root2 > 0 and root1 < 0:
                root = root2
                solution = u + root2 * v + self.RXs[self.ref_index].position
            elif root1 > 0 and root2 > 0:
                solution1 = u + root1 * v + self.RXs[self.ref_index].position
                solution2 = u + root2 * v + self.RXs[self.ref_index].position
                # Choose the solution that minimizes the MSE
                if self.mse_function(solution1) < self.mse_function(solution2):
                    root = root1
                    solution = solution1
                else:
                    root = root2
                    solution = solution2
            else:
                raise ValueError("No valid solution found")
            
            # Update weights based on residuals
            residuals = np.zeros(len(self.tdoas))
            for j, tdoa in enumerate(self.tdoas):
                residuals[j] = dist(solution, tdoa.Rx1.position)
                
            W = np.diag(1 / (residuals**2 + 1e-6))  # Add small value to avoid division by zero
            residuals_record[i] = residuals

        return solution, residuals_record



    def detect_outliers(self, estimate):

        errors = np.zeros(len(self.RXs))
        for j, tdoa in enumerate(self.tdoas):
            predicted_distance = dist(estimate, tdoa.Rx1.position) - dist(estimate, tdoa.Rx2.position)
            error = np.abs(predicted_distance - tdoa.tdoa * c_0)
            errors[j] = error

        return errors
    

    def localize_iterative_outlier_removal(self, iter=2):

        estimate = self.localize_least_squares()

        for i in range(iter):
            errors = self.detect_outliers(estimate)
            print(f"Iteration {i+1}, errors: {errors}")
            # Invalidate receiver with highest error
            max_error_index = np.argmax(errors)
            self.valid[max_error_index] = False  # Invalidate the delay for this receiver
            self.get_tdoas(self.ref_index)  # Recalculate TDOAs with updated receivers
            estimate = self.localize_least_squares()  # Re-localize with updated TDOAs

        return estimate


    def filter_receivers(self, estimate):

        estimated_distances = []
        for i in range(len(self.RXs)):
            estimated_distances.append(dist(estimate, self.RXs[i].position))
        
        C = np.zeros((len(self.RXs), len(self.RXs)))
        k = 0.01
        for i in range(len(self.first_toas)):
            for j in range(i+1, len(self.first_toas)):
                if self.first_toas[i] > 0 and self.first_toas[j] > 0:
                    delta_ij = np.abs(np.log(estimated_distances[i] * self.amplitudes[i] - \
                                             estimated_distances[j] * self.amplitudes[j]))
                    # print(delta_ij)
                    C[i, j] = np.exp(-k * delta_ij**2)

        # print("Consistency scores:", C)
            
        scores = np.zeros(len(self.RXs))
        for i in range(len(self.RXs)):
            for j in range(len(self.RXs)):
                if i != j:
                    scores[i] += C[i, j] 

        # Filter out receivers with low scores
        mean_score = np.mean(scores)
        for i in range(len(self.RXs)):
            if scores[i] < mean_score * 0.5:
                self.valid[i] = False  # Invalidate the delay for this receiver

        print(self.valid)

        return scores

