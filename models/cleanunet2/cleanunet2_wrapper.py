# models/cleanunet2/cleanunet2_wrapper.py

import sys
from pathlib import Path
import torch

# Add the CleanUNet2 directory to Python path so it can find util.py and other dependencies
cleanunet2_dir = Path(__file__).parent
sys.path.insert(0, str(cleanunet2_dir))

from core.base_model import BaseModel
# Now import CleanUNet2 - it should be able to find util.py
from models.cleanunet2.models.cleanunet2 import CleanUNet2

class CleanUNet2Wrapper(BaseModel):
    """Wrapper for CleanUNet2 model"""

    def __init__(self, model_class, config):
        # Fixed constructor to match BaseModel signature
        super().__init__(model_class, config)
        
    def _create_model(self, model_class):
        """Override BaseModel's _create_model to use custom parameter extraction"""
        model_params = self._extract_model_params(self.config)
        return model_class(**model_params)

    def _extract_model_params(self, config):
        net_cfg = config["network_config"]

        return {
            'cleanunet_input_channels': net_cfg.get('cleanunet_input_channels', 1),
            'cleanunet_output_channels': net_cfg.get('cleanunet_output_channels', 1),
            'cleanunet_channels_H': net_cfg.get('cleanunet_channels_H', 32),
            'cleanunet_max_H': net_cfg.get('cleanunet_max_H', 256),
            'cleanunet_encoder_n_layers': net_cfg.get('cleanunet_encoder_n_layers', 5),
            'cleanunet_kernel_size': net_cfg.get('cleanunet_kernel_size', 4),
            'cleanunet_stride': net_cfg.get('cleanunet_stride', 2),
            'cleanunet_tsfm_n_layers': net_cfg.get('cleanunet_tsfm_n_layers', 2),
            'cleanunet_tsfm_n_head': net_cfg.get('cleanunet_tsfm_n_head', 4),
            'cleanunet_tsfm_d_model': net_cfg.get('cleanunet_tsfm_d_model', 128),
            'cleanunet_tsfm_d_inner': net_cfg.get('cleanunet_tsfm_d_inner', 512),
            'cleanspecnet_input_channels': net_cfg.get('cleanspecnet_input_channels', 513),
            'cleanspecnet_num_conv_layers': net_cfg.get('cleanspecnet_num_conv_layers', 3),
            'cleanspecnet_kernel_size': net_cfg.get('cleanspecnet_kernel_size', 3),
            'cleanspecnet_stride': net_cfg.get('cleanspecnet_stride', 1),
            'cleanspecnet_num_attention_layers': net_cfg.get('cleanspecnet_num_attention_layers', 2),
            'cleanspecnet_num_heads': net_cfg.get('cleanspecnet_num_heads', 4),
            'cleanspecnet_hidden_dim': net_cfg.get('cleanspecnet_hidden_dim', 128),
            'cleanspecnet_dropout': net_cfg.get('cleanspecnet_dropout', 0.1),
        }

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        
        # CRITICAL FIX: Clip the audio output to reasonable range
        if isinstance(output, tuple):
            # CleanUNet2 returns (enhanced_audio, enhanced_spec)
            audio, spec = output
            # Clip audio to [-1, 1] range to prevent exploding values
            audio_clipped = torch.clamp(audio, -1.0, 1.0)
            return audio_clipped, spec
        else:
            # If model returns only audio
            return torch.clamp(output, -1.0, 1.0)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
