"""
FIG6 hearing loss compensation module
"""

import numpy as np
import torch

def fig6_prescriptive_formula(audiogram, freq):
    """
    Implements the FIG6 hearing aid fitting formula as described by Killion
    "The 3 types of sensorineural hearing loss: Loudness and intelligibility considerations"
    
    Args:
        audiogram: Hearing thresholds in dB HL at [250, 500, 1000, 2000, 4000, 8000] Hz
        freq: Frequency in Hz to calculate the gain for
        
    Returns:
        Prescribed gain in dB for the given frequency
    """
    # Interpolate the audiogram to get the threshold at the specified frequency
    freqs = np.array([250, 500, 1000, 2000, 4000, 8000])
    if freq <= freqs[0]:
        threshold = audiogram[0]
    elif freq >= freqs[-1]:
        threshold = audiogram[-1]
    else:
        # Find the two closest frequencies
        idx = np.searchsorted(freqs, freq)
        f1, f2 = freqs[idx-1], freqs[idx]
        t1, t2 = audiogram[idx-1], audiogram[idx]
        
        # Linear interpolation in log-frequency domain
        log_f1, log_f2 = np.log10(f1), np.log10(f2)
        log_freq = np.log10(freq)
        weight = (log_freq - log_f1) / (log_f2 - log_f1)
        threshold = t1 + weight * (t2 - t1)
    
    # FIG6 formula:
    # - 0 dB gain for thresholds below 20 dB HL
    # - 1/3 gain for mild loss (20-40 dB)
    # - 1/2 gain for moderate loss (40-60 dB)
    # - 2/3 gain for severe loss (60+ dB)
    
    if threshold <= 20:
        gain = 0
    elif threshold <= 40:
        gain = (threshold - 20) / 3
    elif threshold <= 60:
        gain = (threshold - 40) / 2 + (40 - 20) / 3
    else:
        gain = (threshold - 60) * 2/3 + (60 - 40) / 2 + (40 - 20) / 3
    
    return gain

def apply_fig6_compensation(signal, audiogram, fs=16000):
    """
    Apply FIG6 hearing loss compensation to a signal
    
    Args:
        signal: Input audio signal
        audiogram: Hearing thresholds in dB HL at [250, 500, 1000, 2000, 4000, 8000] Hz
        fs: Sampling rate
        
    Returns:
        Compensated audio signal
    """
    # Check if signal is a tensor
    is_tensor = isinstance(signal, torch.Tensor)
    device = signal.device if is_tensor else None
    
    # Convert to numpy if tensor
    if is_tensor:
        signal_np = signal.cpu().detach().numpy()
    else:
        signal_np = signal
    
    # Get signal length
    N = len(signal_np)
    
    # Apply FFT
    X = np.fft.rfft(signal_np)
    
    # Get frequency bins
    freq_bins = np.fft.rfftfreq(N, d=1/fs)
    
    # Apply frequency-dependent gain
    for i, freq in enumerate(freq_bins):
        # Calculate gain using FIG6 formula
        gain_db = fig6_prescriptive_formula(audiogram, freq)
        
        # Convert from dB to linear scale
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        X[i] = X[i] * gain_linear
    
    # Inverse FFT
    compensated = np.fft.irfft(X, n=N)
    
    # Convert back to tensor if needed
    if is_tensor:
        compensated = torch.tensor(compensated, device=device)
    
    return compensated

def apply_fig6_to_spectrogram(magnitude, audiogram, fs=16000, n_fft=512):
    """
    Apply FIG6 hearing loss compensation to a magnitude spectrogram
    
    Args:
        magnitude: Magnitude spectrogram [B, T, F] or [T, F]
        audiogram: Hearing thresholds in dB HL at [250, 500, 1000, 2000, 4000, 8000] Hz
        fs: Sampling rate
        n_fft: FFT size
        
    Returns:
        Compensated magnitude spectrogram
    """
    # Check if magnitude is a tensor
    is_tensor = isinstance(magnitude, torch.Tensor)
    device = magnitude.device if is_tensor else None
    
    # Get frequency values for each bin
    freq_bins = np.fft.rfftfreq(n_fft, d=1/fs)
    
    # Compute gain for each frequency bin
    gains = []
    for freq in freq_bins:
        gain_db = fig6_prescriptive_formula(audiogram, freq)
        gain_linear = 10 ** (gain_db / 20)
        gains.append(gain_linear)
    gains = np.array(gains)
    
    # Convert to numpy if tensor
    if is_tensor:
        magnitude_np = magnitude.cpu().detach().numpy()
    else:
        magnitude_np = magnitude
    
    # Apply gain based on dimensionality
    if len(magnitude_np.shape) == 3:  # [B, T, F]
        batch_size, time_frames, freq_bins = magnitude_np.shape
        compensated = np.zeros_like(magnitude_np)
        for b in range(batch_size):
            for t in range(time_frames):
                compensated[b, t, :] = magnitude_np[b, t, :] * gains[:freq_bins]
    else:  # [T, F]
        time_frames, freq_bins = magnitude_np.shape
        compensated = np.zeros_like(magnitude_np)
        for t in range(time_frames):
            compensated[t, :] = magnitude_np[t, :] * gains[:freq_bins]
    
    # Convert back to tensor if needed
    if is_tensor:
        compensated = torch.tensor(compensated, device=device)
    
    return compensated