import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


###############################
###### Utility functions ######
###############################

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def get_position_TX(position_RX, delay, angle):
    
    x_RX, y_RX, z_RX = position_RX
    distance = delay * 3e8  
    position_TX = [x_RX + distance * np.cos(angle), y_RX + distance * np.sin(angle), z_RX]
    return position_TX


def find_position_TX(positions_RX, delays, angles):

    x_RX, y_RX, z_RX = positions_RX

    # Select first path
    first_delays = []
    first_angles = []
    for delay, angle in zip(delays, angles):
        index = np.argmin(delay)
        first_delays.append(delay[index])
        first_angles.append(angle[index])


def select_delay(delays, magnitudes):

    energies = magnitudes**2

    
##################################
##### TDOA-based localization ####
##################################


def get_tdoas(delays):
    """
    Calculate TDOAs from the delays of the paths.
    """

    # Set first delay as reference
    reference_delay = 0
    reference_index = 0
    for i in range(len(delays)):
        if delays[i] > 0:
            reference_index = i
            reference_delay = delays[i]
            break

    tdoas = np.zeros(len(delays))
    valid = np.zeros(len(delays), dtype=bool)
    for i in range(reference_index+1, len(delays)):
        if delays[i] > 0:
            tdoa = delays[i] - reference_delay
            tdoas[i] = tdoa
            valid[i] = True
    return tdoas, valid


def mse_function(guess, positions_RX, tdoas, valid):
    cost = 0
    index = 0
    for i in range(len(positions_RX)):
        for j in range(i+1, len(positions_RX)):
            distance = dist(guess, positions_RX[i]) - dist(guess, positions_RX[j])
            cost += (tdoas[index]*3e8 - distance)**2
            index += 1
    return cost


def get_position_from_tdoas(positions_RX, tdoas):

    # Average of RX positions as first guess
    starter_x = 0
    starter_y = 0
    for i in range(len(positions_RX)):
        starter_x += positions_RX[i][0]
        starter_y += positions_RX[i][1]
    starter_x /= len(positions_RX)
    starter_y /= len(positions_RX)

    point = optimize.minimize(fun=mse_function, x0=[starter_x, starter_y], args=(positions_RX, tdoas), method='Nelder-Mead')

    return point.x

        

