# models/unet/loss.py
"""
EXACT loss function from U-Net_speech_enhancement.py
"""
import torch
import torch.nn as nn

class LSDLoss(nn.Module):
    """
    EXACT copy of LSDLoss from your U-Net_speech_enhancement.py
    Log-Spectral Distance Loss
    """
    def __init__(self):
        super(LSDLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        # Calculate squared difference
        squared_diff = torch.mean((y_true - y_pred) ** 2, dim=3)
        # Calculate LSD
        lsd = torch.mean(torch.sqrt(squared_diff), dim=2)
        # Return mean LSD
        return torch.mean(lsd)
