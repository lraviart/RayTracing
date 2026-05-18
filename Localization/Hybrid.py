import numpy as np
from scipy.optimize import minimize, least_squares
from util import global_angle
from Tdoa import Rx, Tdoa

c_0 = 299792458  # Speed of light in m/s

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


class HybridLocalization:


    def __init__(self, Rx_positions, Rx_orientations, toas, aoas, ref_index=None):

        # For TDOA
        self.Rxs = []
        for i in range(len(Rx_positions)):
            self.Rxs.append(Rx(f"Rx{i}", Rx_positions[i]))
        # For Hybrid
        self.Rx_positions = Rx_positions
        self.Rx_orientations = Rx_orientations

        self.valid = np.zeros(len(Rx_positions), dtype=bool)
        self.first_used = np.zeros(len(Rx_positions), dtype=bool)
        self.first_toas = np.zeros(len(Rx_positions))
        self.first_aoas = np.zeros(len(Rx_positions))
        self.second_toas = [[] for _ in range(len(Rx_positions))]
        self.second_aoas = [[] for _ in range(len(Rx_positions))]
        for i in range(len(Rx_positions)):
            if len(toas[i]) > 0:
                self.valid[i] = True
                self.first_used[i] = True
                self.first_toas[i] = toas[i][0]
                self.first_aoas[i] = global_angle(aoas[i][0], self.Rx_orientations[i])
                self.second_toas[i] = toas[i][1:]
                self.second_aoas[i] = global_angle(aoas[i][1:], self.Rx_orientations[i])

        # Find reference delay
        if ref_index is not None:
            if not self.valid[ref_index]:
                raise ValueError("Reference index must have a valid delay")
            self.ref_index = ref_index
            
        else:
            self.ref_index = -1
            for i in range(len(self.first_toas)):
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
                tdoas.append(Tdoa(self.Rxs[i], self.Rxs[ref_index],
                                  self.first_toas[i], self.first_toas[ref_index]))
        self.tdoas = tdoas

        return tdoas

    
    def initial_guess(self, time=False):

        # Initial guess: mean of Rx positions with valid angles and delays
        # Calculate the mean position
        guess = 0
        for i in range(len(self.Rxs)):
            if self.valid[i]:
                guess += self.Rxs[i].position
        guess = guess / np.sum(self.valid)

        if time:
            t0 = np.min([self.first_toas[i] for i in range(len(self.first_toas)) if self.valid[i]])
            guess = np.append(guess, t0)

        return guess
    

    ### First order reflection methods

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
    

    #######################
    ### Error functions ###
    #######################


    ### Measurement error functions ###

    def error_toa(self, estimate, Rx_idx):

        x, y, t0 = estimate
        error = ((self.first_toas[Rx_idx] - t0) * c_0 - dist([x, y], self.Rx_positions[Rx_idx]))
    
        return error
    
    def error_toa_jacobian(self, estimate, Rx_idx):

        x, y, t0 = estimate
        d = dist([x, y], self.Rx_positions[Rx_idx])
        jacobian = np.zeros(3)
        if d == 0:
             jacobian[0] = 0
             jacobian[1] = 0
        else:
            jacobian[0] = -(x - self.Rx_positions[Rx_idx][0]) / d
            jacobian[1] = -(y - self.Rx_positions[Rx_idx][1]) / d
        jacobian[2] = -c_0

        return jacobian
    

    def error_aoa(self, estimate, Rx_idx):

        x, y, _ = estimate
        error = ((x - self.Rx_positions[Rx_idx][0]) * np.sin(self.first_aoas[Rx_idx]) - \
                (y - self.Rx_positions[Rx_idx][1]) * np.cos(self.first_aoas[Rx_idx]))
        return error

    def error_aoa_jacobian(self, estimate, Rx_idx):

        x, y, _ = estimate
        jacobian = np.zeros(3)
        jacobian[0] = np.sin(self.first_aoas[Rx_idx])
        jacobian[1] = -np.cos(self.first_aoas[Rx_idx])
        jacobian[2] = 0

        return jacobian


    def error_hybrid(self, estimate, Rx_idx):

        error = self.error_toa(estimate, Rx_idx)**2 + self.error_aoa(estimate, Rx_idx)**2
        
        return error
    

    def error_hybrid_point(self, estimate, Rx_idx):

        x, y, t0 = estimate
        
        toa = self.first_toas[Rx_idx] - t0
        point = self.Rx_positions[Rx_idx] + \
                toa * c_0 * np.array([np.cos(self.first_aoas[Rx_idx]), 
                                        np.sin(self.first_aoas[Rx_idx])])
        error = np.linalg.norm(point - np.array([x, y]))**2

        return error
    

    def error_hybrid_second(self, estimate, prior, Rx_idx, toa, aoa):

        x, y, t0 = estimate

        # Identify wall
        direct_aoa = np.arctan2(prior[1] - self.Rx_positions[Rx_idx][1], prior[0] - self.Rx_positions[Rx_idx][0])
        wall = self.identify_wall(direct_aoa, aoa)

        # Compute error 
        error = 0
        delta_t = toa - t0
        if wall == 'vertical':
            error += (self.Rx_positions[Rx_idx][1] + delta_t * c_0 * np.sin(aoa) - y)
        elif wall == 'horizontal':
            error += (self.Rx_positions[Rx_idx][0] + delta_t * c_0 * np.cos(aoa) - x)

        return error
    

    def error_hybrid_second_jacobian(self, estimate, prior, Rx_idx, toa, aoa):

        x, y, t0 = estimate

        # Identify wall
        direct_aoa = np.arctan2(prior[1] - self.Rx_positions[Rx_idx][1], prior[0] - self.Rx_positions[Rx_idx][0])
        wall = self.identify_wall(direct_aoa, aoa)

        jacobian = np.zeros(3)
        delta_t = toa - t0
        if wall == 'vertical':
            jacobian[0] = 0
            jacobian[1] = -1
            jacobian[2] = -c_0 * np.sin(aoa)
        elif wall == 'horizontal':
            jacobian[0] = -1
            jacobian[1] = 0
            jacobian[2] = -c_0 * np.cos(aoa)

        return jacobian
    
    
    ### Direct path methods ###

    def mse_tdoa_star(self, estimate):
        # MSE with 1 reference node (star configuration)
        cost = 0
        for tdoa in self.tdoas:
            distance = dist(estimate, tdoa.Rx1.position) - dist(estimate, tdoa.Rx2.position)
            cost += (tdoa.tdoa * 3e8 - distance)**2

        for i, aoa in enumerate(self.first_aoas):
            if self.valid[i]:
                cost += ((estimate[0] - self.Rx_positions[i][0]) * np.sin(aoa) - \
                     (estimate[1] - self.Rx_positions[i][1]) * np.cos(aoa))**2

        return cost
    

    def mse_tdoa_chain(self, estimate):
        # MSE with chain configuration
        x, y = estimate
        cost = 0
        for i in range(len(self.Rx_positions)-1):
            if self.valid[i]:
                distance = dist(estimate, self.Rx_positions[i]) - dist(estimate, self.Rx_positions[(i+1) % len(self.Rx_positions)])
                cost += ((self.first_toas[i] - self.first_toas[(i+1) % len(self.first_toas)]) * c_0 - distance)**2
        for i, aoa in enumerate(self.first_aoas):
            if self.valid[i]:
                cost += ((x - self.Rx_positions[i][0]) * np.sin(aoa) - \
                     (y - self.Rx_positions[i][1]) * np.cos(aoa))**2

        return cost

    
    def mse(self, estimate, idx=None):

        cost = 0

        if idx is None:
             for i in range(len(self.Rx_positions)):
                if self.valid[i]:
                    cost += self.error_hybrid(estimate, i)

        else:
            for i in range(len(self.Rx_positions)):
                if idx[i]:
                    cost += self.error_hybrid(estimate, i)

        return cost
    

    def residuals(self, estimate, idx=None):

        res = []

        if idx is None:
             for i in range(len(self.Rx_positions)):
                if self.valid[i]:
                    res.append(self.error_toa(estimate, i))
                    res.append(self.error_aoa(estimate, i))

        else:
            for i in range(len(self.Rx_positions)):
                if idx[i]:
                    res.append(self.error_toa(estimate, i))
                    res.append(self.error_aoa(estimate, i))

        return np.array(res)
    

    def residuals_jacobian(self, estimate, idx=None):

        x, y, t0 = estimate
        jacobian = []

        if idx is None:
             for i in range(len(self.Rx_positions)):
                if self.valid[i]:
                    jacobian.append(self.error_toa_jacobian(estimate, i))
                    jacobian.append(self.error_aoa_jacobian(estimate, i))

        else:
            for i in range(len(self.Rx_positions)):
                if idx[i]:
                    jacobian.append(self.error_toa_jacobian(estimate, i))
                    jacobian.append(self.error_aoa_jacobian(estimate, i))

        return np.array(jacobian)
    

    ### First order reflection methods ###

    def mse_second(self, estimate, prior, idx=None):

        x, y, t0 = estimate
        cost = 0
        
        if idx is None:
            for i in range(len(self.Rx_positions)):
                if self.valid[i]:
                    if not self.first_used[i]: # Use first tap as if it is a second path
                        cost += self.error_hybrid_second(estimate, prior, i, self.first_toas[i], self.first_aoas[i])**2

                    for j in range(np.minimum(len(self.second_toas[i]), 2)):
                        cost += self.error_hybrid_second(estimate, prior, i, self.second_toas[i][j], self.second_aoas[i][j])**2

        else:
            for i in range(len(self.Rx_positions)):
                if self.valid[i]:
                    if not self.first_used[i]: # Use first tap as if it is a second path
                        cost += self.error_hybrid_second(estimate, prior, i, self.first_toas[i], self.first_aoas[i])**2

                    for j in range(np.minimum(len(self.second_toas[i]), 2)):
                        if idx[i][j]: # Select second path based on RANSAC inlier mask
                            cost += self.error_hybrid_second(estimate, prior, i, self.second_toas[i][j], self.second_aoas[i][j])**2

        return cost
    

    def residuals_second(self, estimate, prior, idx=None):

        x, y, t0 = estimate
        res = []

        if idx is None:
            for i in range(len(self.Rx_positions)):
                if self.valid[i]:
                    if not self.first_used[i]: # Use first tap as if it is a second path
                        res.append(self.error_hybrid_second(estimate, prior, i, self.first_toas[i], self.first_aoas[i]))

                    for j in range(np.minimum(len(self.second_toas[i]), 2)):
                        res.append(self.error_hybrid_second(estimate, prior, i, self.second_toas[i][j], self.second_aoas[i][j]))

        else:
            for i in range(len(self.Rx_positions)):
                if self.valid[i]:
                    for j in range(len(idx[i])):
                        if idx[i][j]: # Select second path based on RANSAC inlier mask
                            if j == 0: # Use first tap as if it is a second path
                                res.append(self.error_hybrid_second(estimate, prior, i, self.first_toas[i], self.first_aoas[i]))
                            else:
                                res.append(self.error_hybrid_second(estimate, prior, i, self.second_toas[i][j-1], self.second_aoas[i][j-1]))
        return np.array(res)
    

    def residuals_second_jacobian(self, estimate, prior, idx=None):

        x, y, t0 = estimate
        jacobian = []

        if idx is None:
            for i in range(len(self.Rx_positions)):
                if self.valid[i]:
                    if not self.first_used[i]: # Use first tap as if it is a second path
                        jacobian.append(self.error_hybrid_second_jacobian(estimate, prior, i, self.first_toas[i], self.first_aoas[i]))

                    for j in range(np.minimum(len(self.second_toas[i]), 2)):
                        jacobian.append(self.error_hybrid_second_jacobian(estimate, prior, i, self.second_toas[i][j], self.second_aoas[i][j]))

        else:
            for i in range(len(self.Rx_positions)):
                if self.valid[i]:
                    for j in range(len(idx[i])):
                        if idx[i][j]: # Select second path based on RANSAC inlier mask
                            if j == 0: # Use first tap as if it is a second path
                                jacobian.append(self.error_hybrid_second_jacobian(estimate, prior, i, self.first_toas[i], self.first_aoas[i]))
                            else:
                                jacobian.append(self.error_hybrid_second_jacobian(estimate, prior, i, self.second_toas[i][j-1], self.second_aoas[i][j-1]))
        return np.array(jacobian)
    

    ############################
    ### Localization methods ###
    ############################

    def localize(self):

        point = minimize(self.mse_tdoa_star, self.initial_guess(), method='Nelder-Mead')

        return point.x
    

    def localize_time(self, initial_guess=None):

        if initial_guess is None:
            initial_guess = self.initial_guess(time=True)

        point = least_squares(lambda x: self.residuals(x), initial_guess, 
                              method='lm', jac=lambda x: self.residuals_jacobian(x))

        return point.x[:2], point.x[2]
    

    def localize_with_second(self):

        prior, t0 = self.localize_time()

        point = minimize(lambda x: self.mse(x) + self.mse_second(x), np.append(prior, t0), method='Nelder-Mead')

        return point.x
    

    def localize_iterative(self, max_iterations=10, tolerance=1e-6):

        estimate, t0 = self.localize_time()
        estimate = np.append(estimate, t0)

        for _ in range(max_iterations):
            estimate = minimize(lambda x:  self.mse(x) + self.mse_second(x), estimate, method='Nelder-Mead').x
            if np.linalg.norm(estimate[:2] - estimate[:2]) < tolerance:
                break

        return estimate
    


    def localize_ransac(self, iter=10, threshold=1, final_output=True):

        best_estimate = None
        best_nb_inliers = -1
        best_inliers = np.zeros_like(self.valid, dtype=bool)

        # RANSAC
        subset_size = 2
        for _ in range(iter):
            # Randomly select a subset of valid paths
            inliers = np.zeros_like(self.valid, dtype=bool)
            indices = np.where(self.valid)[0]
            if len(indices) < subset_size:
                raise ValueError("Not enough valid paths for RANSAC")
            sample_indices = np.random.choice(indices, size=subset_size, replace=False)
            inliers[sample_indices] = True
            sample_estimate = least_squares(lambda x: self.residuals(x, idx=inliers), self.initial_guess(time=True), method='lm', jac=lambda x: self.residuals_jacobian(x, idx=inliers)).x

            # Count inliers
            nb_inliers = 0
            
            for i in indices:
                if self.error_hybrid_point(sample_estimate, i) < threshold:
                    inliers[i] = True
                    nb_inliers += 1

            # Update best estimate
            if nb_inliers > best_nb_inliers:
                best_nb_inliers = nb_inliers
                best_inliers = inliers.copy()
                best_estimate = sample_estimate

        # Recompute estimate using all inliers
        best_estimate = least_squares(lambda x: self.residuals(x, idx=best_inliers), best_estimate, 
                                      method='lm', jac=lambda x: self.residuals_jacobian(x, idx=best_inliers)).x

        if final_output:
            return best_estimate[:2]
        
        else:
            return best_estimate, best_inliers
    


    def localize_all(self, iter=100, threshold=0.5):

        # First estimate using RANSAC
        first_estimate, first_inliers = self.localize_ransac(iter=iter, threshold=threshold, final_output=False)

        # Use second paths
        initial_guess = first_estimate.copy()
        
        second_inliers = [np.array([False]*(len(self.second_toas[i])+1), dtype=bool) for i in range(len(self.Rx_positions))]
        second_presence = False
        for i in range(len(self.Rx_positions)):
            if self.valid[i]:
                for j in range(len(self.second_toas[i])+1):
                    if not (j == 0 and first_inliers[i]):
                        if j == 0: # Use first tap as if it is a second path
                            error = self.error_hybrid_second(first_estimate, initial_guess, i, self.first_toas[i], self.first_aoas[i])**2
                        else:
                            error = self.error_hybrid_second(first_estimate, initial_guess, i, self.second_toas[i][j-1], self.second_aoas[i][j-1])**2
                        
                        if error < threshold/2:
                            second_inliers[i][j] = True
                            second_presence = True
                        
        # Final estimate using all inliers
        if second_presence:
            best_estimate = least_squares(lambda x: np.concatenate([self.residuals(x, idx=first_inliers), 2*self.residuals_second(x, initial_guess, idx=second_inliers)], axis=0), initial_guess, 
                                      method='lm', jac=lambda x: np.concatenate([self.residuals_jacobian(x, idx=first_inliers), 2*self.residuals_second_jacobian(x, initial_guess, idx=second_inliers)], axis=0)).x
        else:
            best_estimate = least_squares(lambda x: self.residuals(x, idx=first_inliers), initial_guess, 
                                      method='lm', jac=lambda x: self.residuals_jacobian(x, idx=first_inliers)).x

        return best_estimate[:2]
    


    def localize_iterative2(self, iter=10, ransac_iter=50, threshold=0.5):

        # First estimate using RANSAC
        first_estimate, first_inliers = self.localize_ransac(iter=ransac_iter, threshold=threshold, final_output=False)

        # Iteratively refine estimate using RANSAC with second paths
        best_estimate = first_estimate
        subset_size = 4
        tol = 1e-5

        # Each candidate is a tuple: (rx_index, path_type, path_index, num_residuals)
        # path_type: 1 for first path (LOS), 2 for second path (NLOS)
        path_pool = []
        for i in range(len(self.Rx_positions)):
            if self.valid[i]:
                # 1. The first tap treated as a first path (LOS, yields 2 residuals)
                path_pool.append((i, 1, 0, 2))
                
                # 2. The first tap treated as a second path (NLOS, yields 1 residual)
                # (You mapped this to index 0 in second_inliers)
                path_pool.append((i, 2, 0, 1))
                
                # 3. The actual subsequent taps treated as second paths
                for j in range(len(self.second_toas[i])):
                    # j+1 because index 0 is reserved for the first tap above
                    path_pool.append((i, 2, j + 1, 1))

        for _ in range(iter):
            ransac_estimate = None
            ransac_nb_inliers = -1
            ransac_cost = np.inf
            ransac_first_inliers = None
            ransac_second_inliers = None
            ransac_first_presence = False
            ransac_second_presence = False
            for _ in range(ransac_iter):
                first_inliers = np.array([False]*len(self.valid), dtype=bool)
                second_inliers = [np.array([False]*(len(self.second_toas[i])+1), dtype=bool) for i in range(len(self.Rx_positions))]
                first_presence = False
                second_presence = False

                # Randomly select a subset of first and second paths
                # nb_paths = len(self.valid) + sum([len(self.second_toas[i]) for i in range(len(self.Rx_positions)) if self.valid[i]])
                # indices = np.arange(nb_paths)
                # indices = np.delete(indices, np.where(~self.valid)[0]) # Remove invalid first paths
                # if len(indices) < subset_size:
                #     print(self.valid)
                #     print("Not enough valid paths for RANSAC iteration")
                #     return best_estimate[:2]
                # sample_indices = np.random.choice(indices, size=subset_size, replace=False)
                # for idx in sample_indices:
                #     if idx < len(self.valid):
                #         if self.valid[idx]:
                #             first_inliers[idx] = True
                #             first_presence = True
                #     else:
                #         idx -= len(self.valid)
                #         for i in range(len(self.Rx_positions)):
                #             if self.valid[i]:
                #                 if idx < len(self.second_toas[i]):
                #                     second_inliers[i][idx] = True
                #                     second_presence = True
                #                     break
                #                 else:
                #                     idx -= len(self.second_toas[i])

                # --- STRICT LOGIC: Construct the sample step-by-step ---
                
                # Check if we have enough independent paths to begin with
                # Each valid receiver provides 1 independent first tap + N second taps
                independent_paths_available = sum(1 + len(self.second_toas[i]) for i in range(len(self.Rx_positions)) if self.valid[i])
                if independent_paths_available < subset_size:
                    print("Not enough independent valid paths for RANSAC iteration")
                    return best_estimate[:2]

                # Create a fresh copy of the pool for this specific iteration
                available_paths = path_pool.copy()
                sample = []
                
                # Draw exactly 'subset_size' paths one by one
                for _ in range(subset_size):
                    # Pick a random index from whatever is currently available
                    chosen_idx = np.random.choice(len(available_paths))
                    
                    # Remove it from the available pool and add to our sample
                    chosen_path = available_paths.pop(chosen_idx)
                    sample.append(chosen_path)
                    
                    # STRICT MUTUAL EXCLUSION: 
                    # If we just picked a first tap (pidx == 0), we must find and destroy 
                    # its mutually exclusive counterpart so it can't be picked next.
                    rx, ptype, pidx, _ = chosen_path
                    if pidx == 0:
                        # If we picked type 1 (LOS), the conflict is type 2 (NLOS). And vice versa.
                        conflict_type = 2 if ptype == 1 else 1
                        
                        # Rebuild the available pool without the conflicting path
                        available_paths = [
                            p for p in available_paths 
                            if not (p[0] == rx and p[1] == conflict_type and p[2] == 0)
                        ]

                # --- Map the strict sample back to your inlier arrays ---
                first_inliers = np.zeros(len(self.valid), dtype=bool)
                second_inliers = [np.zeros(len(self.second_toas[i]) + 1, dtype=bool) for i in range(len(self.Rx_positions))]
                first_presence = False
                second_presence = False
                
                for rx, ptype, pidx, _ in sample:
                    if ptype == 1:
                        first_inliers[rx] = True
                        first_presence = True
                    else:
                        second_inliers[rx][pidx] = True
                        second_presence = True


                # Compute estimate using selected paths
                for i in range(len(self.Rx_positions)):
                    if first_presence and second_presence:
                        estimate = least_squares(lambda x: np.concatenate([self.residuals(x, idx=first_inliers), self.residuals_second(x, best_estimate, idx=second_inliers)], axis=0), best_estimate, 
                                        method='lm', jac=lambda x: np.concatenate([self.residuals_jacobian(x, idx=first_inliers), self.residuals_second_jacobian(x, best_estimate, idx=second_inliers)], axis=0)).x
                    elif first_presence:
                        estimate = least_squares(lambda x: self.residuals(x, idx=first_inliers), best_estimate, 
                                        method='lm', jac=lambda x: self.residuals_jacobian(x, idx=first_inliers)).x
                    elif second_presence:
                        estimate = least_squares(lambda x: self.residuals_second(x, best_estimate, idx=second_inliers), best_estimate, 
                                        method='lm', jac=lambda x: self.residuals_second_jacobian(x, best_estimate, idx=second_inliers)).x
                        
                # Count inliers
                nb_inliers = 0
                inliers_first = np.zeros_like(self.valid, dtype=bool)
                inliers_second = [np.array([False]*(len(self.second_toas[i])+1), dtype=bool) for i in range(len(self.Rx_positions))]
                inliers_first_presence = False
                inliers_second_presence = False
                for i in range(len(self.Rx_positions)):
                    if self.valid[i]:
                        error = self.error_hybrid_point(estimate, i)**2
                        if error < threshold:
                            inliers_first[i] = True
                            inliers_first_presence = True
                            nb_inliers += 1

                        for j in range(len(self.second_toas[i])+1):
                            if j == 0: # Use first tap as if it is a second path
                                error = self.error_hybrid_second(estimate, best_estimate, i, self.first_toas[i], self.first_aoas[i])**2
                            else:
                                error = self.error_hybrid_second(estimate, best_estimate, i, self.second_toas[i][j-1], self.second_aoas[i][j-1])**2
                            if error < threshold and not (j == 0 and inliers_first[i]):
                                inliers_second[i][j] = True
                                inliers_second_presence = True
                                nb_inliers += 1
                inliers_cost = (self.mse(estimate, idx=inliers_first) + self.mse_second(estimate, best_estimate, idx=inliers_second)) # / (nb_inliers + sum(inliers_first)) 
                            
                # Update best estimate
                if nb_inliers > ransac_nb_inliers: 
                    ransac_estimate = estimate.copy()
                    ransac_nb_inliers = nb_inliers
                    ransac_cost = inliers_cost
                    ransac_first_inliers = inliers_first.copy()
                    ransac_second_inliers = [inliers_second[i].copy() for i in range(len(self.Rx_positions))]
                    ransac_first_presence = inliers_first_presence
                    ransac_second_presence = inliers_second_presence
                elif nb_inliers == ransac_nb_inliers and inliers_cost < ransac_cost:
                    ransac_estimate = estimate.copy()
                    ransac_cost = inliers_cost
                    ransac_first_inliers = inliers_first.copy()
                    ransac_second_inliers = [inliers_second[i].copy() for i in range(len(self.Rx_positions))]
                    ransac_first_presence = inliers_first_presence
                    ransac_second_presence = inliers_second_presence

            # Final estimate using all inliers
            print("Ransac iteration: nb_inliers =", ransac_nb_inliers)
            final_estimate = None
            if ransac_first_presence and ransac_second_presence and ransac_nb_inliers >= 2:
                final_estimate = least_squares(lambda x: np.concatenate([self.residuals(x, idx=ransac_first_inliers), self.residuals_second(x, ransac_estimate, idx=ransac_second_inliers)], axis=0), ransac_estimate, 
                                        method='lm', jac=lambda x: np.concatenate([self.residuals_jacobian(x, idx=ransac_first_inliers), self.residuals_second_jacobian(x, ransac_estimate, idx=ransac_second_inliers)], axis=0)).x
            elif ransac_first_presence and ransac_nb_inliers >= 2:
                final_estimate = least_squares(lambda x: self.residuals(x, idx=ransac_first_inliers), ransac_estimate, 
                                        method='lm', jac=lambda x: self.residuals_jacobian(x, idx=ransac_first_inliers)).x
            elif ransac_second_presence and ransac_nb_inliers >= 3:
                final_estimate = least_squares(lambda x: self.residuals_second(x, ransac_estimate, idx=ransac_second_inliers), ransac_estimate, 
                                        method='lm', jac=lambda x: self.residuals_second_jacobian(x, ransac_estimate, idx=ransac_second_inliers)).x

            if final_estimate is not None and np.linalg.norm(final_estimate[:2] - best_estimate[:2]) > tol:
                best_estimate = final_estimate
            else:
                break

            # threshold *= 0.75  # Decrease threshold for next iteration

        return best_estimate[:2]
    

    def localize_iterative(self, iter=100, threshold=0.5):

        # First estimate using RANSAC
        first_estimate, first_inliers = self.localize_ransac(iter=iter, threshold=threshold, final_output=False)

        # Use second paths
        initial_guess = first_estimate.copy()
        second_inliers = [np.array([False]*(len(self.second_toas[i])+1), dtype=bool) for i in range(len(self.Rx_positions))]

        best_estimate = first_estimate.copy()
        tol = 1e-2
        for _ in range(10):
            first_presence = False
            second_presence = False
            for i in range(len(self.Rx_positions)):
                if self.valid[i]:
                    for j in range(len(self.second_toas[i])+1):
                        if j == 0 and first_inliers[i]:
                            error = self.error_hybrid_point(best_estimate, i)**2
                            if error > threshold:
                                first_inliers[i] = False
                            else:
                                first_presence = True

                        if not (j == 0 and first_inliers[i]):
                            if j == 0: # Use first tap as if it is a second path
                                error = self.error_hybrid_second(best_estimate, best_estimate, i, self.first_toas[i], self.first_aoas[i])**2
                            else:
                                error = self.error_hybrid_second(best_estimate, best_estimate, i, self.second_toas[i][j-1], self.second_aoas[i][j-1])**2
                            
                            if error < threshold:
                                second_inliers[i][j] = True
                                second_presence = True
                            
            # Final estimate using all inliers
            if first_presence and second_presence:
                final_estimate = least_squares(lambda x: np.concatenate([self.residuals(x, idx=first_inliers), self.residuals_second(x, initial_guess, idx=second_inliers)], axis=0), initial_guess, 
                                        method='lm', jac=lambda x: np.concatenate([self.residuals_jacobian(x, idx=first_inliers), self.residuals_second_jacobian(x, initial_guess, idx=second_inliers)], axis=0)).x
            elif first_presence:
                final_estimate = least_squares(lambda x: self.residuals(x, idx=first_inliers), initial_guess, 
                                        method='lm', jac=lambda x: self.residuals_jacobian(x, idx=first_inliers)).x
            elif second_presence:
                final_estimate = least_squares(lambda x: self.residuals_second(x, initial_guess, idx=second_inliers), initial_guess, 
                                        method='lm', jac=lambda x: self.residuals_second_jacobian(x, initial_guess, idx=second_inliers)).x
            else:
                break

            if np.linalg.norm(final_estimate[:2] - best_estimate[:2]) > tol:
                best_estimate = final_estimate
            else:
                break

        return best_estimate[:2]



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