import torch
import numpy as np

def extend_audiogram(audiogram, n_freq_bins=257):
    """
    Extend audiogram along frequency axis to match spectrogram dimensions
    Based on Table I in the paper
    
    Args:
        audiogram: Tensor of shape [6] with hearing loss thresholds at 
                   [250, 500, 1000, 2000, 4000, 8000] Hz
        n_freq_bins: Number of frequency bins in the STFT
        
    Returns:
        Extended audiogram of shape [n_freq_bins]
    """
    # Create empty extended audiogram
    device = audiogram.device
    extended = torch.zeros(n_freq_bins, device=device)
    
    # Map audiogram values to frequency bins based on Table I in the paper
    # These ranges are for 512-point FFT with 16kHz sample rate
    extended[0:8] = audiogram[0]      # 0-250 Hz
    extended[8:16] = audiogram[1]     # 250-500 Hz
    extended[16:32] = audiogram[2]    # 500-1000 Hz
    extended[32:64] = audiogram[3]    # 1000-2000 Hz
    extended[64:128] = audiogram[4]   # 2000-4000 Hz
    extended[128:] = audiogram[5]     # 4000-8000 Hz
    
    # Normalize to range [0, 1] for easier processing
    extended = extended / 120.0  # Assuming max hearing loss is 120 dB
    
    return extended

def fig6_compensation(clean_mag, audiogram_extended):
    """
    Apply FIG6 formula to compensate for hearing loss
    This is a simplified implementation - replace with full implementation later
    
    Args:
        clean_mag: Clean speech magnitude spectrogram
        audiogram_extended: Extended audiogram
        
    Returns:
        Compensated clean speech magnitude spectrogram
    """
    # Ensure shapes are compatible
    if audiogram_extended.dim() == 3:  # [B, 1, F]
        # For [B, F, T] input
        gain = audiogram_extended.permute(0, 2, 1)  # [B, F, 1]
    else:  # [B, 1, 1, F]
        # For [B, 1, F, T] input
        gain = audiogram_extended.permute(0, 3, 1, 2)  # [B, F, 1, 1]
    
    # Convert normalized values [0, 1] back to dB
    gain_db = gain * 120.0
    
    # Apply 1/3 compensation ratio as a simple approximation of FIG6
    # Real FIG6 implementation would consider more factors
    compensation_ratio = 1/3
    applied_gain_db = gain_db * compensation_ratio
    
    # Convert from dB to linear scale
    gain_linear = torch.pow(10, applied_gain_db / 20.0)
    
    # Apply gain to clean magnitude
    compensated = clean_mag * gain_linear
    
    return compensated

def load_audiograms(file_path):
    """
    Load audiograms from file
    
    Args:
        file_path: Path to audiogram file
        
    Returns:
        Dictionary mapping file IDs to audiogram thresholds
    """
    audiograms = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:  # Need file ID + 6 frequency thresholds
                    continue
                    
                file_id = parts[0]
                # Convert audiogram thresholds to floats
                thresholds = [float(x) for x in parts[1:7]]
                audiograms[file_id] = thresholds
    except Exception as e:
        print(f"Error loading audiograms: {e}")
    
    return audiograms
