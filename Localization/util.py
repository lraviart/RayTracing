import numpy as np


def format_cir(cir):

    """
    Extract the delays and magnitudes from the cir object given by sionna
    
    Args: 
        cir: tuple of (coeffs_re, coeffs_im), delays
        
    Returns:
        delays: list of arrays (for each receiver) of delays of the paths
        magnitudes: list of arrays (for each receiver) of magnitudes of the paths"""

    coeffs_re, coeffs_im = cir[0]
    delays = cir[1]

    # To numpy arrays
    coeffs_re = coeffs_re[..., 0].numpy() # Number of time steps = 1
    coeffs_im = coeffs_im[..., 0].numpy()
    delays = delays[..., :].numpy()

    magnitudes = np.abs(coeffs_re + 1j*coeffs_im)
    magnitudes = magnitudes.reshape(magnitudes.shape[0], -1)
    delays = delays.reshape(delays.shape[0], -1)

    # Removing points corresponding to -1 delay (no path)
    mag_ = magnitudes
    del_ = delays
    magnitudes = []
    delays = []
    for i in range(mag_.shape[0]):
        mask = del_[i] >= 0
        delays.append(del_[i][mask])
        magnitudes.append(mag_[i][mask])

    # Sort paths by delay
    for i in range(len(delays)):
        sort_indices = np.argsort(delays[i])
        delays[i] = delays[i][sort_indices]
        magnitudes[i] = magnitudes[i][sort_indices]

    return delays, magnitudes