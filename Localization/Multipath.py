import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.signal import resample, find_peaks
from scipy.ndimage import maximum_filter
from numpy.lib.stride_tricks import sliding_window_view
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


### 1D MUSIC algorithm ###

def MUSIC_spectrum_OFDM(B, fft_size, num_ofdm_symbols, a, tau, no=1e-12):

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


def MUSIC_spectrum(a, tau, B, fft_size=512, T=1e-6, Tcir=2e-7, osr=8, alpha=0.1, num_pilots=10):
    
    fft_size_ = int(fft_size * (1-alpha))
    h_hat = np.zeros((num_pilots, fft_size_), dtype=complex)
    for i in range(num_pilots):
        h_time = com.get_sc_cp_channel_response(a, tau, B, osr, T=T, Tcir=Tcir, fft_size=fft_size, alpha=alpha)[0]
        h_hat[i] = np.fft.fftshift(np.fft.fft(h_time))[int(fft_size*alpha/2):int(fft_size*alpha/2)+fft_size_]

    # Deconvolution 
    # frequencies = np.linspace(-B/2, B/2, fft_size)
    # h_hat = h_hat / com.RC_freq(frequencies, B, alpha)

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
    # print("Eigenvalues:", eigenvalues[k_opt-5:k_opt+5])
    # k_opt = 4

    # Noise subspace
    U = eigenvectors[:, k_opt:]

    # MUSIC spectrum
    tau = np.linspace(0, Tcir, 1000)
    freq = np.arange(M) * (B/fft_size) 

    # Broadcasting
    tau_grid = tau[None, :]                 # Shape: (1, len(tau))
    f_grid = freq[:, None]                  # Shape: (M, 1)
    a = np.exp(-1j * 2*np.pi * f_grid * tau_grid) # Shape: (M, len(tau))
    P = 1 / np.sum(np.abs(U.conj().T @ a)**2, axis=0)

    return P, tau, k_opt, h_hat


def MUSIC_taps_OFDM(spectrum, tau, df, L, h_hat):

    peaks, _ = find_peaks(spectrum)
    peaks = peaks[np.argsort(spectrum[peaks])][::-1][:L] # Select top L peaks

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


def MUSIC_taps(spectrum, tau, df, K, h_hat):
    
    peaks, _ = find_peaks(spectrum)
    peaks = peaks[np.argsort(spectrum[peaks])][::-1][:K] # Select top K peaks

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



### 2D FFT algorithm ###

def FFT_2D_spectrum(a, tau, B, fc=3.5e9, fft_size=512, T=1e-6, Tcir=2e-7, osr=8, alpha=0.1, num_pilots=10, pad_angle=512, pad_tau=1024):
    
    # ---------------------------------------------------------
    # 1. Data Generation (Identical to your original code)
    # ---------------------------------------------------------
    fft_size_ = int((fft_size) * (1-alpha)) # Truncation to avoid aliasing
    num_rx_ant = a.shape[0]
    h_hat = np.zeros((num_pilots, num_rx_ant, fft_size_), dtype=complex)
    
    for i in range(num_pilots):
        for j in range(num_rx_ant):
            # Assuming 'com' is imported and available in your scope
            h_time = com.get_sc_cp_channel_response(a[j], tau, B, osr, T=T, Tcir=Tcir, fft_size=fft_size, alpha=alpha)[0]
            h_hat[i, j] = np.fft.fftshift(np.fft.fft(h_time))[int(fft_size*alpha/2):int(fft_size*alpha/2)+fft_size_]

    # ---------------------------------------------------------
    # 2. Data Preparation for FFT
    # ---------------------------------------------------------
    # Average across the pilots (snapshots) to improve SNR
    h_mean = np.mean(h_hat, axis=0) # Shape: (num_rx_ant, fft_size_)
    
    # Apply a 2D Window (Hanning) to heavily suppress Rayleigh sidelobes
    win_ant = np.hanning(num_rx_ant)
    win_freq = np.hanning(fft_size_)
    h_windowed = h_mean * win_ant[:, None] * win_freq[None, :]

    # ---------------------------------------------------------
    # 3. 2D Transform
    # ---------------------------------------------------------
    # Step A: IFFT across frequency (axis=1) to transition to Delay (ToA)
    h_ifft = np.fft.ifft(h_windowed, n=pad_tau, axis=1)
    
    # Step B: FFT across antennas (axis=0) to extract Spatial Freq (AoA)
    h_fft_2d = np.fft.fft(h_ifft, n=pad_angle, axis=0)
    
    # Shift zero-angle to the center of the array
    h_fft_2d_shifted = np.fft.fftshift(h_fft_2d, axes=0)
    
    # Compute the power spectrum heatmap
    P = np.abs(h_fft_2d_shifted)**2

    # ---------------------------------------------------------
    # 4. Axis Mapping (Bins to Physical Values)
    # ---------------------------------------------------------
    # --- Delay (Tau) Mapping ---
    Delta_f = B / fft_size  # Subcarrier spacing
    dt = 1.0 / (pad_tau * Delta_f)
    tau_array = np.arange(pad_tau) * dt
    
    # Truncate to Tcir to match the grid bounds of your MUSIC code
    valid_tau = tau_array <= Tcir
    if not np.any(valid_tau): # Fallback if Tcir is extremely small
        valid_tau = np.ones(pad_tau, dtype=bool)
        
    tau_array = tau_array[valid_tau]
    P = P[:, valid_tau]
    
    # --- Angle Mapping ---
    # Shifted indices from -pad_angle/2 to pad_angle/2 - 1
    m = np.arange(-pad_angle//2, pad_angle//2)
    
    # Based on your implicit MUSIC steering vector, element spacing is lambda/2.
    # spatial_freq = pi * sin(theta) => sin(theta) = 2 * m / pad_angle
    sin_theta = 2.0 * m / pad_angle
    
    # Filter out the "invisible region" where |sin(theta)| > 1
    valid_angle = np.abs(sin_theta) <= 1.0
    sin_theta = sin_theta[valid_angle]
    angles_array = np.arcsin(sin_theta) # Bounds correctly from -pi/2 to pi/2
    
    P = P[valid_angle, :]
    
    # Return P.T so shape matches your original MUSIC output (len(tau), len(angles)).
    # We return None for k_opt since FFT doesn't calculate MDL paths.
    return P.T, tau_array, angles_array, None, h_hat



### 2D MUSIC algorithm ###

def MUSIC_2D_spectrum(a, tau, B, fc=3.5e9, fft_size=512, T=1e-6, Tcir=2e-7, osr=8, alpha=0.1, num_pilots=10):

    fft_size_ = int((fft_size) * (1-alpha)) # Truncation to avoid the aliasing part
    num_rx_ant = a.shape[0]
    h_hat = np.zeros((num_pilots, num_rx_ant, fft_size_), dtype=complex)
    for i in range(num_pilots):
        for j in range(num_rx_ant):
            h_time = com.get_sc_cp_channel_response(a[j], tau, B, osr, T=T, Tcir=Tcir, fft_size=fft_size, alpha=alpha)[0]
            h_hat[i, j] = np.fft.fftshift(np.fft.fft(h_time))[int(fft_size*alpha/2):int(fft_size*alpha/2)+fft_size_]

    # print("com part done")

    ### MUSIC algorithm
    M = fft_size_ // 2 # Size of subblocks
    N_sym = fft_size_ - M + 1
    N = num_pilots * N_sym # Total snapshots
    
    # Smoothing
    windows = sliding_window_view(h_hat, window_shape=M, axis=2) # (num_pilots, num_rx_ant, N_sym, M)  
    windows = np.swapaxes(windows, 1, 2) # (num_pilots, N_sym, num_rx_ant, M)
    h_blocks = windows.reshape(num_pilots * N_sym, num_rx_ant * M)

    # Autocorreleation matrix
    R = h_blocks.T @ h_blocks.conj() / N
    # print("R shape:", R.shape)

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
    # print("Optimal number of paths:", k_opt)
    # print("Eigenvalues around the threshold:", eigenvalues[k_opt-3:k_opt+5])

    # Noise subspace
    U = eigenvectors[:, k_opt:]

    # MUSIC spectrum
    tau = np.linspace(0, Tcir, 200)
    angles = np.linspace(-np.pi, np.pi, 180)
    freq = np.arange(M) * (B/fft_size)

    if k_opt == 0: # No paths detected, return empty spectrum
        return np.zeros((len(angles), len(tau))), tau, angles, k_opt, h_hat

    # Broadcasting
    tau_grid = tau[None, :, None, None]                 # Shape: (1, len(tau), 1, 1)
    angles_grid = angles[:, None, None, None]           # Shape: (len(angles), 1, 1, 1)
    k_grid = np.arange(num_rx_ant)[None, None, :, None] # Shape: (1, 1, num_rx_ant, 1)
    f_grid = freq[None, None, None, :]                  # Shape: (1, 1, 1, M)

    # Steering vector tensor (len(angles), len(tau), num_rx_ant, M)
    a_tensor = np.exp(-1j * 2 * np.pi * f_grid * tau_grid + 1j * np.pi * np.sin(angles_grid) * k_grid)
    a = a_tensor.reshape(len(angles) * len(tau), num_rx_ant * M)

    # Projection of steering vectors onto the noise subspace
    proj = np.sum(np.abs(a @ U.conj())**2, axis=1) # (len(angles) * len(tau), L-k_opt)
    
    # Calculate P and reshape it back to the 2D grid shape
    P = (1 / proj).reshape(len(angles), len(tau)).T

    return P, tau, angles, k_opt, h_hat


def find_2D_peaks(spectrum, num_peaks):
    
    if num_peaks == 0:
        return np.array([])
    
    safe_spectrum = spectrum + np.random.uniform(0, 1e-10, size=spectrum.shape)

    # Find local maxima in the 2D spectrum
    footprint = np.ones((5, 5))
    mask = maximum_filter(safe_spectrum, footprint=footprint, mode='wrap') == safe_spectrum
    peaks = np.argwhere(mask)

    # Select top peaks
    peak_values = safe_spectrum[mask]
    top_peaks = peaks[np.argsort(peak_values)[::-1][:num_peaks]]

    return top_peaks



def MUSIC_2D_taps(spectrum, tau, angles, df, K, h_hat):

    num_rx_ant = h_hat.shape[1]
    fft_size_ = h_hat.shape[2]

    peaks = find_2D_peaks(spectrum, num_peaks=K)
    estimated_taus = [tau[peaks[i][0]] for i in range(K)]
    estimated_angles = [angles[peaks[i][1]] for i in range(K)]

    sort_idx = np.argsort(estimated_taus)
    estimated_taus = np.array(estimated_taus)[sort_idx]
    estimated_angles = np.array(estimated_angles)[sort_idx]

    # Average channel estimates
    h_mean = np.mean(h_hat, axis=0).flatten()
    
    # Construct steering matrix A
    frequencies = np.arange(fft_size_) * df

    # Broadcasting
    f_grid = frequencies[None, :, None]
    rx_ant_grid = np.arange(num_rx_ant)[:, None, None]
    tau_grid = estimated_taus[None, None, :]
    angle_grid = estimated_angles[None, None, :]

    # Steering matrix
    A_tensor = np.exp(-1j * 2*np.pi * f_grid * tau_grid + 1j * np.pi * np.sin(angle_grid) * rx_ant_grid)
    A = A_tensor.reshape(num_rx_ant * fft_size_, K)

    # Solve linear system A * a = h_mean using Least Squares
    a_hat, residuals, rank, s = np.linalg.lstsq(A, h_mean, rcond=None)
    
    return a_hat, estimated_taus, estimated_angles

















