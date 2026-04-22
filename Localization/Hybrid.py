import numpy as np
from scipy import optimize
from Tdoa import Rx, Tdoa
from Aoa import Aoa

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

class HybridLocalization:

    def __init__(self, toas, aoas, RX_positions):

        self.RXs = []
        for i in range(len(RX_positions)):
            self.RXs.append(Rx(f"Rx{i}", RX_positions[i]))

        self.tdoas = []
        
        # Find reference delay
        ref_index = -1
        for i in range(len(toas)):
            if toas[i] > 0:
                ref_index = i
                break
        if ref_index == -1:
            raise ValueError("No valid delay found")
        
        # Calculate TDOAs
        for i in range(ref_index+1, len(toas)):
            if toas[i] > 0:
                # print(i)
                self.tdoas.append(Tdoa(self.RXs[ref_index], self.RXs[i], toas[ref_index], toas[i]))

        self.aoas = []
        for i in range(len(aoas)):
            if aoas[i] < 2*np.pi:
                self.aoas.append(Aoa(self.RXs[i], aoas[i]))

    
    def initial_guess(self):

        # Initial guess: mean of Rx positions with valid angles and delays
        valid_RXs = []
        for tdoa in self.tdoas:
            if tdoa.Rx1 not in valid_RXs:
                valid_RXs.append(tdoa.Rx1)
            if tdoa.Rx2 not in valid_RXs:
                valid_RXs.append(tdoa.Rx2)

        for aoa in self.aoas:
            if aoa.RX not in valid_RXs:
                valid_RXs.append(aoa.RX)

        # Calculate the mean position
        guess = np.mean([rx.position for rx in valid_RXs], axis=0)

        return guess
    

    def mse(self, estimate):

        cost = 0
        for tdoa in self.tdoas:
            distance = dist(estimate, tdoa.Rx1.position) - dist(estimate, tdoa.Rx2.position)
            cost += (tdoa.tdoa * 3e8 - distance)**2

        for aoa in self.aoas:
            cost += aoa.cost(estimate)**2

        return cost
    

    def localize(self):

        point = optimize.minimize(self.mse, self.initial_guess(), method='Nelder-Mead')

        return point.x