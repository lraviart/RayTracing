import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.signal import resample, find_peaks
import Comm as com


###############################
###### Utility functions ######
###############################

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


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
  


###################################
######## MUSIC algorithm ##########
###################################


### MUSIC algorithm

def MUSIC_spectrum(B, fft_size, num_ofdm_symbols, a, tau, no=0, time_domain=False):


    if not time_domain:

        ### Simulation of the OFDM system
        frequencies = com.subcarrier_frequencies(fft_size, B/fft_size)
        h = com.cir_to_ofdm_channel(frequencies, a, tau)

        x = com.pilot_pattern(num_ofdm_symbols, fft_size)
        x = (1+1j) * np.ones_like(x) # For simplicity, use all ones as pilots
        y = com.apply_ofdm_channel(x, h, no)

        # Estimation of the channel
        h_hat = com.estimate_channel(y, x)

        ### MUSIC algorithm

        # Subdivision of the channel into subblocks
        M = fft_size // 2 # Size of subblocks
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
        # k_opt = 20

        # Noise subspace
        U = eigenvectors[:, k_opt:]

        # MUSIC spectrum
        tau = np.linspace(0, 2e-7, 1000)
        freq = np.arange(M) * (B/fft_size) 
        a = np.exp(-1j * 2 * np.pi * np.outer(freq, tau))
        P = 1 / np.sum(np.abs(U.conj().T @ a)**2, axis=0)


    ### Time domain approach 
    if time_domain:

        t = np.linspace(0, fft_size/B, fft_size, endpoint=False)
        h_time = np.zeros(fft_size, dtype=complex)
        for i in range(len(a)):
            h_time += a[i] * com.RC(t, tau[i], B, alpha=0.5) # Using Raised Cosine pulse for time domain representation

        # Deconvolution with the pulse shape (assuming perfect knowledge of the pulse)
        h_freq = np.fft.fft(h_time)
        pulse_freq = np.fft.fft(com.RC(t, 0, B, alpha=0.5))
        h_deconvolved = h_freq / pulse_freq
    
        
        # Add noise
        h_hat = np.zeros((num_ofdm_symbols, h_time.shape[0]), dtype=complex)
        for i in range(num_ofdm_symbols):
            noise = np.random.normal(size=h_time.shape) + 1j * np.random.normal(size=h_time.shape)
            h_hat[i] = h_deconvolved + no * noise

        ### MUSIC algorithm

        # Subdivision of the channel into subblocks
        M = fft_size // 2 # Size of subblocks
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
        # k_opt = 20

        # Noise subspace
        U = eigenvectors[:, k_opt:]

        # MUSIC spectrum
        tau = np.linspace(0, 2e-7, 1000)
        freq = np.arange(M) * (B/fft_size) 
        a = np.exp(-1j * 2 * np.pi * np.outer(freq, tau))
        P = 1 / np.sum(np.abs(U.conj().T @ a)**2, axis=0)

    return P, tau, h_hat


def MUSIC_spectrum_time(a, tau, B, osr, num_symbols):
    
    
    h_hat, t, _ = com.get_channel_response(a, tau, B, osr=osr, N=1000, alpha=0.5)


    h_blocks = np.zeros()

    ### MUSIC algorithm

    # Subdivision of the channel into subblocks
    M = fft_size // 2 # Size of subblocks
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
    # k_opt = 20

    # Noise subspace
    U = eigenvectors[:, k_opt:]

    # MUSIC spectrum
    tau = np.linspace(0, 2e-7, 1000)
    freq = np.arange(M) * (B/fft_size) 
    a = np.exp(-1j * 2 * np.pi * np.outer(freq, tau))
    P = 1 / np.sum(np.abs(U.conj().T @ a)**2, axis=0)

    return P, tau, h_hat


def MUSIC_taps(spectrum, tau, B, fft_size, h_hat):

    peaks, _ = find_peaks(spectrum)
    estimated_taus = tau[peaks]
    
    # Average channel estimates
    h_mean = np.mean(h_hat, axis=0) 
    
    # Construct steering matrix A
    frequencies = com.subcarrier_frequencies(fft_size, B/fft_size)
    A = np.exp(-1j * 2 * np.pi * np.outer(frequencies, estimated_taus))
    
    # Solve linear system A * a = h_mean using Least Squares
    # rcond=None suppresses a warning and uses the machine precision default
    a_hat, residuals, rank, s = np.linalg.lstsq(A, h_mean)
    
    return a_hat, estimated_taus



###############################
##### Main taps estimation ####
###############################


def MSE_taps(x, taps, t, B):

    # Extract parameters
    a_re = x[::3]
    a_im = x[1::3]
    tau = x[2::3]
    
    # Reconstruct CIR from estimated parameters
    a = a_re + 1j * a_im
    h_estimated = np.zeros_like(taps, dtype=complex)
    for i in range(len(tau)):
        h_estimated += a[i] * np.sinc((t - tau[i]) * B)

    # Compute MSE
    mse = np.mean(np.abs(h_estimated - taps)**2)
    
    return mse


def estimate_taps(taps, t, B, N_taps): 

    # Initial guess for optimization
    initial_guess = np.zeros(3 * N_taps)  # [a1_re, a1_im, tau1, a2_re, a2_im, tau2, ..., aN_re, aN_im, tauN]
    for i in range(N_taps):
        initial_guess[3*i] = np.max(np.abs(taps)) * (1 + 0.1 * np.random.randn())  # Initial real amplitude guess
        initial_guess[3*i + 1] = np.max(np.abs(taps)) * 0.1 * np.random.randn()  # Initial imaginary amplitude guess
        initial_guess[3*i + 2] = t[np.argmax(np.abs(taps))] * (1 + 0.1 * np.random.randn())  # Initial delay guess
    print("Initial guess:", initial_guess)


    # Optimize parameters to minimize MSE
    result = optimize.minimize(MSE_taps, initial_guess, args=(taps, t, B), method='Nelder-Mead')

    # Extract estimated parameters
    estimated_a = result.x[::3] + 1j * result.x[1::3]
    estimated_tau = result.x[2::3]

    return estimated_a, estimated_tau