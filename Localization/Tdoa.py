import numpy as np
from scipy import optimize


def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


class Rx:

    def __init__(self, name, position):

        self.name = name
        self.position = position


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
        ref_index = -1
        for i in range(len(first_toas)):
            if first_toas[i] > 0:
                ref_index = i
                break
        if ref_index == -1:
            raise ValueError("No valid delay found")
        
        # Calculate TDOAs
        for i in range(ref_index+1, len(first_toas)):
            if first_toas[i] > 0:
                # print(i)
                self.tdoas.append(Tdoa(self.RXs[ref_index], self.RXs[i], first_toas[ref_index], first_toas[i]))
        


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
            cost += (tdoa.tdoa * 3e8 - distance)**2
        return cost
    

    def localize(self):
        point = optimize.minimize(fun=self.mse_function, x0=self.initial_guess(), args=(), method='Nelder-Mead')

        return point.x