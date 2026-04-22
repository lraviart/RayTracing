import numpy as np


def format_cir(cir):

    """
    Extract the delays and magnitudes from the cir object given by sionna
    
    Args: 
        cir: tuple of (coeffs_re, coeffs_im), delays
        
    Returns:
        delays: list of arrays (for each receiver) of delays of the paths
        magnitudes: list of arrays (for each receiver) of magnitudes of the paths
    
    """

    coeffs_re, coeffs_im = cir[0]
    delays = cir[1]

    # To numpy arrays
    coeffs_re = coeffs_re[..., 0].numpy() # Number of time steps = 1
    coeffs_im = coeffs_im[..., 0].numpy()
    delays = delays[..., :].numpy()

    a = coeffs_re + 1j*coeffs_im
    a = a.reshape(a.shape[0], -1)
    tau = delays.reshape(delays.shape[0], -1)

    # Removing points corresponding to -1 delay (no path)
    a_ = a
    tau_ = tau
    a = []
    tau = []
    for i in range(a_.shape[0]):
        mask = tau_[i] >= 0
        a.append(a_[i][mask])
        tau.append(tau_[i][mask])

    # Sort paths by delay
    for i in range(len(tau)):
        sort_indices = np.argsort(tau[i])
        tau[i] = tau[i][sort_indices]
        a[i] = a[i][sort_indices]

    return a, tau



def format_paths(cir, angles, array=False):

    """
    Extract the delays, magnitudes and angles from the cir and angles objects given by sionna

    Args:
        cir: tuple of (coeffs_re, coeffs_im), delays
        angles: list of arrays of shape [num_rx, num_tx, num_paths] 

    Returns:
        delays: list of arrays (for each receiver) of delays of the paths
        magnitudes: list of arrays (for each receiver) of magnitudes of the paths
        angles: list of arrays (for each receiver) of angles of the paths
    
    """

    coeffs_re, coeffs_im = cir[0]
    delays = cir[1]

    # To numpy arrays
    coeffs_re = coeffs_re[..., 0].numpy() # Number of time steps = 1
    coeffs_im = coeffs_im[..., 0].numpy()
    delays = delays[..., :].numpy()
    angles = angles[..., :].numpy()

    a = coeffs_re + 1j*coeffs_im
    if not array:
        a = a.reshape(a.shape[0], -1) 
    else:
        a = a.reshape(a.shape[0], a.shape[1], -1)
    tau = delays.reshape(delays.shape[0], -1)
    angles = angles.reshape(angles.shape[0], -1)


    # Removing points corresponding to -1 delay (no path)
    a_ = a
    tau_ = tau
    angles_ = angles
    a = []
    tau = []
    angles = []
    for i in range(a_.shape[0]):
        mask = tau_[i] >= 0
        if not array:
            a.append(a_[i][mask])
        else:
            a_append = np.zeros((a_.shape[1], np.sum(mask)), dtype=complex)
            for j in range(a_.shape[1]):
                a_append[j] = a_[i, j][mask]
            a.append(a_append)
        tau.append(tau_[i][mask])
        angles.append(angles_[i][mask])

    # Sort paths by delay
    for i in range(len(tau)):
        sort_indices = np.argsort(tau[i])
        if not array:
            a[i] = a[i][sort_indices]
        else:
            a_append = np.zeros((a[i].shape[0], a[i].shape[1]), dtype=complex)
            for j in range(a_.shape[1]):
                a_append[j] = a[i][j][sort_indices]
            a[i] = a_append
        tau[i] = tau[i][sort_indices]
        angles[i] = angles[i][sort_indices]

    return a, tau, angles
    

def bound_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def format_local_angles(angles, RX_orientations):

    """
    Format the angles given by sionna to be relative to the receiver orientation

    Args:
        angles: list of arrays of shape [num_rx, num_tx, num_paths] 
        RX_orientations: list of arrays of shape [num_rx, 3] (yaw, pitch, roll)

    Returns:
        angles: list of arrays (for each receiver) of angles of the paths relative to the receiver orientation
    
    """
    local_angles = []
    for i in range(len(angles)):
        local_angles.append(bound_angle(angles[i] - RX_orientations[i][0]))

    return local_angles


def global_angle(local_angle, RX_orientation):
    return bound_angle(local_angle + RX_orientation[0])
