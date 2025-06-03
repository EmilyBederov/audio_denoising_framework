# models/unet/models/unet.py
"""
EXACT UNet model implementation from U-Net_speech_enhancement.py
This is the exact UNetModel class that was working in the original code
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    EXACT copy of UNetModel from U-Net_speech_enhancement.py
    """
    def __init__(self):
        super(UNet, self).__init__()
        
        # Simplified encoder (just 3 layers) - EXACT copy
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Simplified decoder (just 3 layers) - EXACT copy
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder - EXACT copy
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Decoder with skip connections - EXACT copy
        d1 = self.dec1(e3)
        d1 = F.interpolate(d1, size=e2.size()[2:], mode='nearest')
        d2 = self.dec2(torch.cat([d1, e2], dim=1))
        d2 = F.interpolate(d2, size=e1.size()[2:], mode='nearest')
        d3 = self.dec3(torch.cat([d2, e1], dim=1))
        d3 = F.interpolate(d3, size=x.size()[2:], mode='nearest')
        
        return d3