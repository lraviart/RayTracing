import numpy as np
from scipy import optimize
from Tdoa import Rx

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    

class Aoa:

    def __init__(self, RX, angle):

        self.RX = RX
        self.angle = angle


    def cost(self, estimate):

        dx = estimate[0] - self.RX.position[0]
        dy = estimate[1] - self.RX.position[1]

        return dx * np.sin(self.angle) - dy * np.cos(self.angle)
       


class AoaLocalization:

    def __init__(self, AoAs, RX_positions):

        self.RXs = []
        for i in range(len(RX_positions)):
            self.RXs.append(Rx(f"Rx{i}", RX_positions[i]))

        self.aoas = []
        for i in range(len(AoAs)):
            if AoAs[i] < 2*np.pi:
                self.aoas.append(Aoa(self.RXs[i], AoAs[i]))



    def initial_guess(self):

        # Initial guess: mean of Rx positions with valid angles
        valid_RXs = []
        for aoa in self.aoas:
            if aoa.RX not in valid_RXs:
                valid_RXs.append(aoa.RX)

        # Calculate the mean position
        guess = np.mean([rx.position for rx in valid_RXs], axis=0)

        return guess
    

    def mse(self, estimate):
        cost = 0
        for aoa in self.aoas:
            cost += aoa.cost(estimate)**2
        return cost
    

    def localize(self):
        point = optimize.minimize(self.mse, self.initial_guess(), method='Nelder-Mead')

        return point.x
    

    def localize_least_squares(self):

        A = np.zeros((len(self.aoas), 2))
        b = np.zeros(len(self.aoas))

        for i, aoa in enumerate(self.aoas):
            A[i, 0] = np.sin(aoa.angle)
            A[i, 1] = -np.cos(aoa.angle)
            b[i] = aoa.RX.position[0] * np.sin(aoa.angle) - aoa.RX.position[1] * np.cos(aoa.angle)

        # Solve the least squares problem
        estimate = np.linalg.lstsq(A, b, rcond=None)[0]

        return estimate