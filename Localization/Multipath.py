import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.signal import resample


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


def clean_algorithm(taps, t):

    max_iter = 100
    threshold = np.max(np.abs(taps)) * 1e-2
    B = 1 / (t[1] - t[0])

    oversample_factor = 10
    num_samples = len(taps)

    taps_oversampled, t_oversampled = resample(taps, num_samples * oversample_factor, t=t)
    original_taps = taps_oversampled.copy()
    cleaned_taps = np.zeros_like(taps_oversampled)

    index = 0
    for _ in range(max_iter):
        max_index = np.argmax(np.abs(taps_oversampled))
        if np.abs(taps_oversampled[max_index]) < threshold:
            break
        cleaned_taps[max_index] += 0.1 * taps_oversampled[max_index]
        taps_oversampled -= taps_oversampled[max_index] * 0.1 * np.sinc((t_oversampled - t_oversampled[max_index]) * B)
        index += 1

    return cleaned_taps, t_oversampled, taps_oversampled, original_taps, index
  




# OFDM framework for MUSIC algorithm

def cir_to_ofdm_channel(frequencies, a, tau):

    h = np.zeros_like(frequencies, dtype=complex)
    for i in range(len(a)):
        h += a[i] * np.exp(-1j * 2 * np.pi * frequencies * tau[i])
    return h


def apply_ofdm_channel(x, h, no):

    noise = np.random.normal(size=x.shape) + 1j * np.random.normal(size=x.shape)
    y = h * x + no * noise    
    return y


def subcarrier_frequencies(fft_size, subcarrier_spacing):

    return np.arange(-fft_size//2, fft_size//2) * subcarrier_spacing


def zadoff_chu_seq(u, N):

    n = np.arange(N)
    seq = np.exp(-1j * np.pi * u * n * (n + 1) / N)
    return seq


def pilot_pattern(num_ofdm_symbols, fft_size):

    # Zadoff-Chu sequence
    u = 23
    x = np.zeros((num_ofdm_symbols, fft_size), dtype=complex)
    for i in range(num_ofdm_symbols):
        x[i] = zadoff_chu_seq(u, fft_size)

    return x


def estimate_channel(y, x):

    h_hat = y / x
    return h_hat


def MUSIC(B, fft_size, num_ofdm_symbols, a, tau, no=0):

    ### Simulation of the OFDM system
    frequencies = subcarrier_frequencies(fft_size, B/fft_size)
    h = cir_to_ofdm_channel(frequencies, a, tau)

    x = pilot_pattern(num_ofdm_symbols, fft_size)
    y = apply_ofdm_channel(x, h, no)

    # Estimation of the channel
    h_hat = estimate_channel(y, x)


    ### MUSIC algorithm

    # Subdivision of the channel into subblocks
    M = 512 # Size of subblocks
    N_sym = fft_size - M + 1 
    N = num_ofdm_symbols * N_sym # Total snapshots
    h_blocks = np.zeros((N, M), dtype=complex)
    
    idx = 0
    for i in range(num_ofdm_symbols):
        for j in range(N_sym):
            h_blocks[idx] = h_hat[i, j:j+M]
            idx += 1
    # Autocorreleation matrix
    R = h_blocks.T @ h_blocks.conj() / N

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    eigenvalues = np.flip(eigenvalues)
    eigenvectors = np.flip(eigenvectors, axis=1)

    # Estimation of the number of paths with the MDL criterion
    L = M 
    mdl = np.zeros(L)

    for k in range(L):
        if k == L - 1:
            mdl[k] = 0.5 * k * (2*L - k) * np.log(N)
            continue
            
        log_num = np.sum(np.log(eigenvalues[k:L])) / (L-k)
        log_den = np.log(np.sum(eigenvalues[k:L]) / (L-k))
        log_Lambda_ratio = log_num - log_den
        
        mdl[k] = -N * (L-k) * log_Lambda_ratio + 0.5 * k * (2*L - k) * np.log(N)

    k_opt = np.argmin(mdl)
    print("Optimal number of paths:", k_opt)
    k_opt = 4 # For testing purposes

    # Noise subspace
    U = eigenvectors[:, k_opt:]

    # MUSIC spectrum
    tau = np.linspace(0, 2e-7, 1000)
    freq = np.arange(M) * (B/fft_size) 
    a = np.exp(-1j * 2 * np.pi * np.outer(freq, tau))
    P = 1 / np.sum(np.abs(U.conj().T @ a)**2, axis=0)

    return P, tau, h, h_hat

