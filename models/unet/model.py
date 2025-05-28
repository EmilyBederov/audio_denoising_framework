# models/unet/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base_model import BaseModel

class UNetModel(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, base_channels=32, depth=3):
        super(UNetModel, self).__init__()
        
        # Store configuration
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth
        
        # Simplified encoder
        self.encoder = nn.ModuleList()
        
        # First layer
        self.encoder.append(nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2)
        ))
        
        # Remaining encoder layers
        for i in range(1, depth):
            input_ch = base_channels * (2**(i-1))
            output_ch = base_channels * (2**i)
            
            if i == depth - 1:  # Last encoder layer
                self.encoder.append(nn.Sequential(
                    nn.Conv2d(input_ch, output_ch, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                    nn.BatchNorm2d(output_ch),
                    nn.LeakyReLU(0.2)
                ))
            else:
                self.encoder.append(nn.Sequential(
                    nn.Conv2d(input_ch, output_ch, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
                    nn.BatchNorm2d(output_ch),
                    nn.LeakyReLU(0.2)
                ))
        
        # Simplified decoder
        self.decoder = nn.ModuleList()
        
        # Decoder layers (in reverse order)
        for i in range(depth-1, 0, -1):
            input_ch = base_channels * (2**i)
            output_ch = base_channels * (2**(i-1))
            
            if i == depth - 1:  # First decoder layer (matches last encoder)
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(input_ch, output_ch, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
                    nn.BatchNorm2d(output_ch),
                    nn.ReLU()
                ))
            else:
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(input_ch * 2, output_ch, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
                    nn.BatchNorm2d(output_ch),
                    nn.ReLU()
                ))
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, output_channels, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder forward pass
        for enc_layer in self.encoder:
            x = enc_layer(x)
            encoder_outputs.append(x)
        
        # Reverse encoder outputs for decoder skip connections
        encoder_outputs = encoder_outputs[:-1][::-1]
        
        # Decoder forward pass with skip connections
        for i, dec_layer in enumerate(self.decoder):
            if i == 0:  # First decoder layer doesn't have skip connection
                x = dec_layer(x)
                # Ensure sizes match before concatenation
                x = F.interpolate(x, size=encoder_outputs[i].size()[2:], mode='nearest')
            else:
                # Concatenate with skip connection
                x = torch.cat([x, encoder_outputs[i-1]], dim=1)
                x = dec_layer(x)
                if i < len(encoder_outputs):
                    # Ensure sizes match before concatenation
                    x = F.interpolate(x, size=encoder_outputs[i].size()[2:], mode='nearest')
        
        # Final layer with last skip connection
        x = torch.cat([x, encoder_outputs[-1]], dim=1)
        x = self.final_layer(x)
        
        # Ensure output size matches input size
        # This is important for audio denoising where output should match input
        if x.size() != encoder_outputs[0].size():
            x = F.interpolate(x, size=encoder_outputs[0].size()[2:], mode='nearest')
        
        return x

class UNet(BaseModel):
    def setup(self):
        """Initialize model based on config"""
        # Extract configuration
        self.input_channels = self.config.get('input_channels', 1)
        self.output_channels = self.config.get('output_channels', 1)
        self.base_channels = self.config.get('base_channels', 32)
        self.depth = self.config.get('depth', 3)
        
        # Initialize UNet model
        self.model = UNetModel(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            base_channels=self.base_channels,
            depth=self.depth
        )
        
        # STFT parameters
        self.n_fft = self.config.get('n_fft', 1024)
        self.hop_length = self.config.get('hop_length', 256)
        self.win_length = self.config.get('win_length', 1024)
        
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def get_loss(self, targets, predictions):
        """Calculate loss for training"""
        # Use log-spectral distance (LSD) loss as in the original code
        from models.unet.loss import LSDLoss
        if not hasattr(self, '_loss_fn'):
            self._loss_fn = LSDLoss()
        return self._loss_fn(predictions, targets)
