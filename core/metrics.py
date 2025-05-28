import numpy as np
import torch
from pystoi import stoi
from pypesq import pesq
from scipy.signal import correlate

def calculate_snr(clean, enhanced):
    """
    Signal-to-Noise Ratio
    clean, enhanced: [B, T] torch tensors
    """
    snrs = []
    for c, e in zip(clean, enhanced):
        c = c.cpu().numpy()
        e = e.cpu().numpy()
        noise = c - e
        snr = 10 * np.log10((np.sum(c**2) + 1e-10) / (np.sum(noise**2) + 1e-10))
        snrs.append(snr)
    return np.mean(snrs)

def calculate_stoi(clean, enhanced, sample_rate):
    """
    Short-Term Objective Intelligibility
    clean, enhanced: [B, T] torch tensors
    """
    stois = []
    for c, e in zip(clean, enhanced):
        c = c.cpu().numpy()
        e = e.cpu().numpy()
        min_len = min(len(c), len(e))
        stois.append(stoi(c[:min_len], e[:min_len], sample_rate, extended=False))
    return np.mean(stois)

def calculate_pesq(clean, enhanced, sample_rate):
    """
    PESQ score (ITU-T P.862)
    clean, enhanced: [B, T] torch tensors
    sample_rate: 8000 or 16000 only
    """
    pesqs = []
    for c, e in zip(clean, enhanced):
        c = c.cpu().numpy()
        e = e.cpu().numpy()
        min_len = min(len(c), len(e))
        try:
            score = pesq(sample_rate, c[:min_len], e[:min_len], 'wb')
            pesqs.append(score)
        except Exception as ex:
            print(f"[PESQ] Error: {ex}")
            continue
    return np.mean(pesqs) if len(pesqs) > 0 else 0.0
