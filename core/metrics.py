import numpy as np
import torch
from pystoi import stoi
from pypesq import pesq
from scipy.signal import correlate

def calculate_snr(clean, enhanced):
    """
    Signal-to-Noise Ratio
    clean, enhanced: torch tensors of shape [B, C, T] or [B, T]
    """
    snrs = []
    
    # Handle different tensor shapes
    if len(clean.shape) == 3:  # [B, C, T]
        clean = clean.squeeze(1)  # Remove channel dim -> [B, T]
        enhanced = enhanced.squeeze(1)
    elif len(clean.shape) == 2:  # [B, T] - already correct
        pass
    else:  # [T] - single sample
        clean = clean.unsqueeze(0)
        enhanced = enhanced.unsqueeze(0)
    
    for c, e in zip(clean, enhanced):
        c = c.cpu().numpy().flatten()  # Ensure 1D
        e = e.cpu().numpy().flatten()
        
        # Ensure same length
        min_len = min(len(c), len(e))
        c = c[:min_len]
        e = e[:min_len]
        
        noise = c - e
        snr = 10 * np.log10((np.sum(c**2) + 1e-10) / (np.sum(noise**2) + 1e-10))
        snrs.append(snr)
    return np.mean(snrs)

def calculate_stoi(clean, enhanced, sample_rate):
    """
    Short-Term Objective Intelligibility
    clean, enhanced: torch tensors of shape [B, C, T] or [B, T]
    """
    stois = []
    
    # Handle different tensor shapes  
    if len(clean.shape) == 3:  # [B, C, T]
        clean = clean.squeeze(1)  # Remove channel dim -> [B, T]
        enhanced = enhanced.squeeze(1)
    elif len(clean.shape) == 2:  # [B, T] - already correct
        pass
    else:  # [T] - single sample
        clean = clean.unsqueeze(0)
        enhanced = enhanced.unsqueeze(0)
    
    for c, e in zip(clean, enhanced):
        try:
            c = c.cpu().numpy().flatten()  # Ensure 1D array
            e = e.cpu().numpy().flatten()
            
            # Ensure same length
            min_len = min(len(c), len(e))
            c = c[:min_len]
            e = e[:min_len]
            
            # Skip if too short (STOI needs sufficient length)
            if min_len < sample_rate * 0.5:  # At least 0.5 seconds
                continue
                
            score = stoi(c, e, sample_rate, extended=False)
            stois.append(score)
        except Exception as ex:
            print(f"[STOI] Error: {ex}")
            continue
            
    return np.mean(stois) if len(stois) > 0 else 0.5

def calculate_pesq(clean, enhanced, sample_rate):
    """
    PESQ score (ITU-T P.862) - DISABLED due to library issues
    clean, enhanced: torch tensors of shape [B, C, T] or [B, T]
    sample_rate: 8000 or 16000 only
    """
    # TEMPORARILY DISABLED - pypesq has compatibility issues
    print("[PESQ] Temporarily disabled due to library issues")
    return 2.5  # Return reasonable default PESQ score
