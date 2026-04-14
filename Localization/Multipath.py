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

def MUSIC_spectrum(B, fft_size, num_ofdm_symbols, a, tau, no=1e-12):

    ### Simulation of the OFDM system
    frequencies = com.subcarrier_frequencies(fft_size, B/fft_size)
    h = com.cir_to_ofdm_channel(frequencies, a, tau)

    x = com.pilot_pattern(num_ofdm_symbols, fft_size)
    x = (1+1j) * np.ones_like(x) # For simplicity, use all ones as pilots
    y = com.apply_ofdm_channel(x, h, no)

    # Estimation of the channel
    h_hat = com.estimate_channel(y, x)

    # Truncate to the central part of the spectrum to avoid edge effects
    h_hat = h_hat[:, int(fft_size*0.1):int(fft_size*0.9)]
    fft_size_ = h_hat.shape[1]
    print(fft_size_)

    ### MUSIC algorithm

    # Subdivision of the channel into subblocks
    M = fft_size_ // 2 # Size of subblocks
    N_sym = fft_size_ - M + 1 
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


def MUSIC_spectrum_time(a, tau, B, fft_size=512, T=1e-6, Tcir=2e-7, osr=8, alpha=0.1, num_pilots=10, h_hat_debug=None):
    
    fft_size_ = int(fft_size * 0.8)
    h_hat = np.zeros((num_pilots, fft_size_), dtype=complex)
    for i in range(num_pilots):
        h_hat[i] = np.fft.fftshift(np.fft.fft(com.get_sc_cp_channel_response(a, tau, B, osr, T=T, Tcir=Tcir, fft_size=fft_size, alpha=alpha)[0]))[int(fft_size*0.1):int(fft_size*0.9)]

    print(fft_size_)
    if h_hat_debug is not None:
        h_hat = h_hat_debug


    ### MUSIC algorithm

    # Subdivision of the channel into subblocks
    M = fft_size_ // 2 # Size of subblocks
    N_sym = fft_size_ - M + 1
    N = num_pilots * N_sym # Total snapshots
    h_blocks = np.zeros((N, M), dtype=complex)
    
    idx = 0
    for i in range(num_pilots):
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
    print("Eigenvalues:", eigenvalues[k_opt-5:k_opt+5])
    # k_opt = 10

    # Noise subspace
    U = eigenvectors[:, k_opt:]

    # MUSIC spectrum
    tau = np.linspace(0, Tcir, 1000)
    freq = np.arange(M) * (B/fft_size) 
    a = np.exp(-1j * 2*np.pi * np.outer(freq, tau))
    print("a shape:", a.shape)
    P = 1 / np.sum(np.abs(U.conj().T @ a)**2, axis=0)

    return P, tau, h_hat


def MUSIC_taps(spectrum, tau, df, h_hat):

    peaks, _ = find_peaks(spectrum)
    estimated_taus = tau[peaks]
    
    # Average channel estimates
    h_mean = np.mean(h_hat, axis=0) 
    
    # Construct steering matrix A
    frequencies = np.arange(len(h_mean)) * df
    A = np.exp(-1j * 2 * np.pi * np.outer(frequencies, estimated_taus))
    
    # Solve linear system A * a = h_mean using Least Squares
    # rcond=None suppresses a warning and uses the machine precision default
    a_hat, residuals, rank, s = np.linalg.lstsq(A, h_mean)
    
    return a_hat, estimated_taus


def MUSIC_taps_time(spectrum, tau, df, h_hat):
    
    peaks, _ = find_peaks(spectrum)
    estimated_taus = tau[peaks]
    
    # Average channel estimates
    h_mean = np.mean(h_hat, axis=0) 
    
    # Construct steering matrix A
    frequencies = np.arange(len(h_mean)) * df
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



### 2D MUSIC algorithm

def MUSIC_2D_spectrum(a, tau, B, fc=3.5e9, fft_size=512, T=1e-6, Tcir=2e-7, osr=8, alpha=0.1, num_pilots=10, h_hat_debug=None):

    fft_size_ = int(fft_size * 0.8)
    num_rx_ant = a.shape[0]
    h_hat = np.zeros((num_pilots, num_rx_ant, fft_size_), dtype=complex)
    for i in range(num_pilots):
        for j in range(num_rx_ant):
            h_hat[i, j] = np.fft.fftshift(np.fft.fft(com.get_sc_cp_channel_response(a[j], tau, B, osr, T=T, Tcir=Tcir, fft_size=fft_size, alpha=alpha)[0]))[int(fft_size*0.1):int(fft_size*0.1)+fft_size_]

    
    ### MUSIC algorithm
    M = fft_size_ // 2 # Size of subblocks
    N_sym = fft_size_ - M + 1
    N = num_pilots * N_sym # Total snapshots
    h_blocks = np.zeros((N, M*num_rx_ant), dtype=complex)

    idx = 0
    for i in range(num_pilots):
        for j in range(N_sym):
            h_blocks[idx] = h_hat[i, :, j:j+M].reshape(-1)
            idx += 1
    # Autocorreleation matrix
    R = h_blocks.T @ h_blocks.conj() / N
    print("R shape:", R.shape)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    eigenvalues = np.flip(eigenvalues)
    eigenvectors = np.flip(eigenvectors, axis=1)

    # Estimation of the number of paths with the MDL criterion
    L = M * num_rx_ant
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

    # Noise subspace
    U = eigenvectors[:, k_opt:]

    # MUSIC spectrum
    tau = np.linspace(0, Tcir, 100)
    angles = np.linspace(-np.pi, np.pi, 40)
    freq = np.arange(M) * (B/fft_size)
    a = np.zeros((len(angles), len(tau), M * num_rx_ant), dtype=complex)
    for i in range(len(angles)):
        for j in range(len(tau)):
            for k in range(num_rx_ant):
                a[i, j, k*M:(k+1)*M] = np.exp(-1j * 2 * np.pi * freq * tau[j] + (1j * np.pi * np.sin(angles[i]) * k))
    P = np.zeros((len(tau), len(angles)), dtype=float)
    for i in range(len(angles)):
        for j in range(len(tau)):
            P[j, i] = 1 / np.sum(np.abs(U.conj().T @ a[i, j])**2)
    print("P shape:", P.shape)

    return P, tau, angles, h_hat
