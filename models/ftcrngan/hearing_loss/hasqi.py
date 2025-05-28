"""
HASQI (Hearing Aid Speech Quality Index) Implementation
"""

import numpy as np
from scipy import signal
import torch

def hasqi_v2(clean, processed, fs=16000, audiogram=None):
    """
    Simplified implementation of the Hearing Aid Speech Quality Index (HASQI) version 2
    Based on: "Evaluating the generalization of the Hearing Aid Speech Quality Index (HASQI)"
    by Kressner et al.
    
    Args:
        clean: Clean reference speech signal
        processed: Processed/enhanced speech signal
        fs: Sampling rate
        audiogram: Hearing thresholds at [250, 500, 1000, 2000, 4000, 8000] Hz (optional)
        
    Returns:
        HASQI score between 0 and 1 (higher is better)
    """
    # This is a simplified placeholder implementation of HASQI
    # For a real implementation, you would need a more detailed model
    
    # In a real application, replace this with actual HASQI calculation
    # For now, we'll use a simple correlation-based metric for testing purposes
    
    # If tensors are provided, convert to numpy
    if isinstance(clean, torch.Tensor):
        clean = clean.detach().cpu().numpy()
    if isinstance(processed, torch.Tensor):
        processed = processed.detach().cpu().numpy()
    
    # Match lengths
    min_len = min(len(clean), len(processed))
    clean = clean[:min_len]
    processed = processed[:min_len]
    
    # Calculate correlation
    correlation = np.corrcoef(clean, processed)[0, 1]
    
    # Convert to a score between 0 and 1
    hasqi_score = (correlation + 1) / 2
    
    # Add a penalty for overall level differences
    level_diff = np.abs(np.mean(clean**2) - np.mean(processed**2))
    level_penalty = 1 / (1 + level_diff)
    
    # Combine for final score
    final_score = 0.8 * hasqi_score + 0.2 * level_penalty
    
    return max(0, min(1, final_score))  # Ensure score is between 0 and 1