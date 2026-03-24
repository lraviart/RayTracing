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



def format_paths(cir, angles):

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
    a = a.reshape(a.shape[0], -1)
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
        a.append(a_[i][mask])
        tau.append(tau_[i][mask])
        angles.append(angles_[i][mask])

    # Sort paths by delay
    for i in range(len(tau)):
        sort_indices = np.argsort(tau[i])
        a[i] = a[i][sort_indices]
        tau[i] = tau[i][sort_indices]
        angles[i] = angles[i][sort_indices]

    return a, tau, angles
    
