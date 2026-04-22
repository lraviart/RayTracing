import numpy as np
import scipy.signal as sig
from scipy.linalg import toeplitz, pinv
import matplotlib.pyplot as plt

#########################
### Utility functions ###
#########################


def RC(t, tau, B, alpha):
    """
    Computes the Raised Cosine pulse.
    t: Time array (can be shifted by delay, e.g., t - tau)
    tau: Delay
    B: Bandwidth
    alpha: Roll-off factor (0 to 1)
    """
    # Prevent divide by zero warning by adding a tiny epsilon
    # Alternatively, you can use np.where to handle the exact singularity
    
    # Handle the pure sinc case first
    if alpha == 0:
        return np.sinc((t - tau) * B) # np.sinc already includes the pi
        
    # Find the singularity points
    singularity_mask = np.isclose(np.abs(t - tau), 1 / (2 * alpha * B), atol=1e-12)
    
    # Calculate the main equation (using a safe time array to avoid warning)
    safe_t = np.where(singularity_mask, 0, t - tau) # Temp replacement
    
    sinc_part = np.sinc(safe_t * B)
    cos_part = np.cos(np.pi * alpha * safe_t * B)
    den_part = 1 - (2 * alpha * safe_t * B)**2
    
    rc_pulse = sinc_part * (cos_part / den_part)
    
    # Fix the singularity points using L'Hopital's limit
    limit_val = (np.pi / 4) * np.sinc(1 / (2 * alpha))
    rc_pulse[singularity_mask] = limit_val

    # Normalize the pulse energy to 1
    # rc_pulse /= np.sqrt(np.sum(rc_pulse**2))
    
    return rc_pulse


def RRC(t, tau, B, alpha):

    """
    Computes the Root Raised Cosine pulse.
    t: Time array (can be shifted by delay, e.g., t - tau)
    tau: Delay
    B: Bandwidth
    alpha: Roll-off factor (0 to 1)
    """
    # Prevent divide by zero warning by adding a tiny epsilon
    # Alternatively, you can use np.where to handle the exact singularity
    
    # Handle the pure sinc case first
    if alpha == 0:
        return np.sinc((t - tau) * B) # np.sinc already includes the pi
        
    # Find the singularity points
    singularity_mask1 = np.isclose(np.abs(t - tau), 0, atol=1e-12)
    singularity_mask2 = np.isclose(np.abs(t - tau), 1 / (4 * alpha * B), atol=1e-12)
    singularity_mask = singularity_mask1 | singularity_mask2
    
    # Calculate the main equation (using a safe time array to avoid warning)
    safe_t = np.where(singularity_mask, 1/B, t - tau) # Temp replacement
    
    num_part = (np.sin(np.pi * (1 - alpha) * safe_t * B) + 4 * alpha * safe_t * B * np.cos(np.pi * (1 + alpha) * safe_t * B))
    den_part = np.pi * safe_t * B * (1 - (4 * alpha * safe_t * B)**2)
    rrc_pulse = num_part / den_part
    
    # Fix the singularity points using L'Hopital's limit
    limit_val1 = (1 + alpha * (4 / np.pi - 1))
    limit_val2 = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
    rrc_pulse[singularity_mask1] = limit_val1
    rrc_pulse[singularity_mask2] = limit_val2

    # Normalize the pulse amplitude to 1
    # rrc_pulse /= np.max(np.abs(rrc_pulse))

    rrc_pulse *= np.hamming(len(rrc_pulse)) # Apply Hamming window to reduce side lobes
    
    return rrc_pulse


def RC_freq(f, B, alpha):
    """
    Computes the Raised Cosine pulse in frequency domain.
    f: Frequency array
    B: Bandwidth
    alpha: Roll-off factor (0 to 1)
    """
    H = np.zeros_like(f, dtype=complex)
    
    # Define the frequency response based on the piecewise conditions
    for i in range(len(f)):
        if np.abs(f[i]) <= (1 - alpha) * B / 2:
            H[i] = 1
        elif (1 - alpha) * B / 2 < np.abs(f[i]) <= (1 + alpha) * B / 2:
            H[i] = 0.5 * (1 + np.cos(np.pi * (np.abs(f[i]) - (1 - alpha) * B / 2) / (alpha * B)))
        else:
            H[i] = 0
            
    return H


def zadoff_chu_seq(u, N):
    n = np.arange(N)
    
    if N % 2 == 0:
        # EVEN length definition
        seq = np.exp(-1j * np.pi * u * (n ** 2) / N)
    else:
        # ODD length definition
        seq = np.exp(-1j * np.pi * u * n * (n + 1) / N)
        
    return seq



####################
### SC framework ###
####################


def cir_to_sc_channel(a, tau, B, t, alpha=0.5):

    h = np.zeros_like(t, dtype=complex)
    for i in range(len(a)):
        h += a[i] * RRC(t, tau[i], B, alpha) # Using Root Raised Cosine pulse for time domain representation
    return h


def estimate_sc_channel(r, x, L):
    """
    Least squares channel estimation using the known pilot sequence.
    r: Received signal after matched filtering
    x: Known pilot sequence
    N_pilot: Number of pilot symbols
    L: Length of the channel impulse response
    """
    N = len(x)

    # Discard the first L-1 samples and keep only the pilot part
    r_truncated = r[L-1:N]

    # Construct the Toeplitz matrix (N-L+1 x L)
    X = toeplitz(x[L-1:N], np.flip(x[:L]))

    # Solve for h using least squares
    h_hat = np.linalg.lstsq(X, r_truncated, rcond=None)[0]

    return h_hat


def get_sc_channel_response(a, tau, B, osr, T=1e-6, Tcir=2e-7, alpha=0.5):

    Fs = osr * B
    Ts = 1 / Fs
    N = int(T / Ts) # Total number of samples
    L = int(Tcir * B) # Length of the channel impulse response in samples
    N_ext = 2 * N
    t = np.linspace(-N*Ts, N*Ts, N_ext, endpoint=False) # Time array centered around zero
    print("L:", L, "N:", N, "N_ext:", N_ext)
    # Channel in time domain, already convolved with the pulse shape
    h_time = cir_to_sc_channel(a, tau, B, t, alpha)

    # Pilots
    pilot_length = 2 * L - 1
    pilot_seq = zadoff_chu_seq(23, pilot_length)

    t0_idx = N # Find the index corresponding to t=0
    x = np.zeros(N_ext, dtype=complex)
    x[t0_idx:t0_idx+pilot_length*osr:osr] = pilot_seq # Place the pilot

    # Convolution with the channel
    y = np.convolve(x, h_time, mode='full')[t0_idx:t0_idx+N_ext]

    # Add noise
    no = 1.38e-23 * 290 * B * osr # Noise power (k*T*B)
    noise = np.random.normal(size=y.shape) + 1j * np.random.normal(size=y.shape)

    y = y + np.sqrt(no) * noise

    print("Noise power:", np.mean(np.abs(np.sqrt(no) * noise)**2))
    noise_filtered = np.convolve(np.sqrt(no) * noise, RRC(t, 0, B, alpha), mode='full')[t0_idx:t0_idx+N] / osr
    print("Filtered noise power:", np.mean(np.abs(noise_filtered)**2))

    # Matched filtering
    u_matched = RRC(t, 0, B, alpha) # Matched filter is the same as the pulse shape
    r = np.convolve(y, u_matched, mode='full')[t0_idx:t0_idx+N_ext] / osr

    # Synchronisation and Resampling to symbol rate 
    r = r[N:] # Remove the buffer time
    r = r[::osr]

    # Channel estimation (using the known pilot)
    h_hat = estimate_sc_channel(r, pilot_seq, L=L)

    return h_hat, t, h_time


#######################
### SC-CP framework ###
#######################

def cir_to_sc_cp_channel(a, tau, B, alpha=0.5):

    t = np.linspace(0, 1/B, 1000) # Time array for pulse shaping
    h = np.zeros_like(t, dtype=complex)
    for i in range(len(a)):
        h += a[i] * RRC(t, tau[i], B, alpha) # Using Raised Cosine pulse for time domain representation
    return h


def estimate_sc_cp_channel(r, x, L):
    """
    Least squares channel estimation using the known pilot sequence.
    r: Received signal after matched filtering
    x: Known pilot sequence
    N_pilot: Number of pilot symbols
    L: Length of the channel impulse response
    """
    # Synchronization 
    N = len(x)
    
    # Discard the first L samples which are the cyclic prefix
    r_truncated = r[L:N+L]

    r_fft = np.fft.fft(r_truncated)
    x_fft = np.fft.fft(x)

    h_hat_fft = r_fft / x_fft
    h_hat = np.fft.ifft(h_hat_fft)

    return h_hat


def get_sc_cp_channel_response(a, tau, B, osr, T=1e6, Tcir=2e-7, fft_size=512, alpha=0.5):

    Fs = osr * B
    Ts = 1 / Fs
    N = int(T / Ts) # Total number of samples
    L = int(Tcir * B) # Length of the channel impulse response in samples
    N_ext = 2 * N
    t = np.linspace(-N*Ts, N*Ts, N_ext, endpoint=False) # Time array centered around zero

    # Channel in time domain, already convolved with the pulse shape
    h_time = cir_to_sc_channel(a, tau, B, t, alpha)

    # Pilots
    pilot_length = fft_size
    pilot_seq = zadoff_chu_seq(23, pilot_length)
    prefixed_pilot_length = pilot_length + L # Account for cyclic prefix
    prefixed_pilot_seq = np.concatenate((pilot_seq[-L:], pilot_seq)) # Add cyclic prefix
    t0_idx = N # Find the index corresponding to t=0
    x = np.zeros(N_ext, dtype=complex)
    x[t0_idx:t0_idx+prefixed_pilot_length*osr:osr] = prefixed_pilot_seq # Place the prefixed pilot


    # Convolution with the channel
    y = np.convolve(x, h_time, mode='full')[t0_idx:t0_idx+N_ext]

    # Add noise
    no = 1.38e-23 * 290 * B * osr # Noise power (k*T*B)
    noise = np.random.normal(size=y.shape) + 1j * np.random.normal(size=y.shape)

    y = y + np.sqrt(no) * noise

    # print("Noise power:", np.mean(np.abs(np.sqrt(no/2) * noise)**2))
    # noise_filtered = np.convolve(np.sqrt(no/2) * noise, RRC(t, 0, B, alpha), mode='full')[t0_idx:t0_idx+N_ext] / osr
    # print("Filtered noise power:", np.mean(np.abs(noise_filtered)**2))

    # Matched filtering
    u_matched = RRC(t, 0, B, alpha) # Matched filter is the same as the pulse shape
    r = np.convolve(y, u_matched, mode='full')[t0_idx:t0_idx+N_ext] / osr

    # Synchronisation and Resampling to symbol rate 
    r = r[N:] # Remove the buffer time
    r = r[::osr]

    # Channel estimation (using the known pilot)
    h_hat = estimate_sc_cp_channel(r, pilot_seq, L=L)

    return h_hat, t, h_time



######################
### OFDM framework ###
######################


def cir_to_ofdm_channel(frequencies, a, tau):

    h = np.zeros_like(frequencies, dtype=complex)
    for i in range(len(a)):
        h += a[i] * np.exp(-1j * 2 * np.pi * frequencies * tau[i])
    return h


def apply_ofdm_channel(x, h, no):

    noise = np.random.normal(size=x.shape) + 1j * np.random.normal(size=x.shape)
    print("Noise power:", np.mean(np.abs(np.sqrt(no/2) * noise)**2))
    y = h * x + np.sqrt(no/2) * noise    
    return y


def subcarrier_frequencies(fft_size, subcarrier_spacing):

    return np.arange(-fft_size//2, fft_size//2) * subcarrier_spacing




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