# models/cleanunet2/cleanunet2_wrapper.py

import sys
from pathlib import Path

# Add the CleanUNet2 directory to Python path so it can find util.py and other dependencies
cleanunet2_dir = Path(__file__).parent
sys.path.insert(0, str(cleanunet2_dir))

from core.base_model import BaseModel
# Import the ORIGINAL CleanUNet2 (with proper paper defaults)
from models.cleanunet2.models.cleanunet2 import CleanUNet2 as OriginalCleanUNet2

class CleanUNet2Wrapper(BaseModel):
    """Wrapper for CleanUNet2 model using the original implementation"""

    def __init__(self, model_class, config):
        # Use the original CleanUNet2 instead of the model_class parameter
        super().__init__(OriginalCleanUNet2, config)
        
    def _create_model(self, model_class):
        """Override BaseModel's _create_model to use original CleanUNet2 parameters"""
        model_params = self._extract_model_params(self.config)
        return OriginalCleanUNet2(**model_params)

    def _extract_model_params(self, config):
        """Extract and map parameters for the original CleanUNet2"""
        net_cfg = config["network_config"]
        
        # Option 1: Use original defaults (recommended)
        if net_cfg.get('use_original_defaults', True):
            print(" Using original CleanUNet2 defaults (paper configuration)")
            return {}  # Empty dict = use all defaults
        
        # Option 2: Custom parameters (for advanced users)
        return {
            # CleanUNet parameters
            'cleanunet_input_channels': net_cfg.get('cleanunet_input_channels', 1),
            'cleanunet_output_channels': net_cfg.get('cleanunet_output_channels', 1),
            'cleanunet_channels_H': net_cfg.get('cleanunet_channels_H', 64),      # Original default
            'cleanunet_max_H': net_cfg.get('cleanunet_max_H', 768),               # Original default
            'cleanunet_encoder_n_layers': net_cfg.get('cleanunet_encoder_n_layers', 8),  # Original default
            'cleanunet_kernel_size': net_cfg.get('cleanunet_kernel_size', 4),
            'cleanunet_stride': net_cfg.get('cleanunet_stride', 2),
            'cleanunet_tsfm_n_layers': net_cfg.get('cleanunet_tsfm_n_layers', 5),     # Original default
            'cleanunet_tsfm_n_head': net_cfg.get('cleanunet_tsfm_n_head', 8),
            'cleanunet_tsfm_d_model': net_cfg.get('cleanunet_tsfm_d_model', 512),     # Original default
            'cleanunet_tsfm_d_inner': net_cfg.get('cleanunet_tsfm_d_inner', 2048),    # Original default
            
            # CleanSpecNet parameters
            'cleanspecnet_input_channels': net_cfg.get('cleanspecnet_input_channels', 513),
            'cleanspecnet_num_conv_layers': net_cfg.get('cleanspecnet_num_conv_layers', 5),     # Original default
            'cleanspecnet_kernel_size': net_cfg.get('cleanspecnet_kernel_size', 4),             # Original default
            'cleanspecnet_stride': net_cfg.get('cleanspecnet_stride', 1),
            'cleanspecnet_num_attention_layers': net_cfg.get('cleanspecnet_num_attention_layers', 5),  # Original default
            'cleanspecnet_num_heads': net_cfg.get('cleanspecnet_num_heads', 8),                 # Original default
            'cleanspecnet_hidden_dim': net_cfg.get('cleanspecnet_hidden_dim', 512),             # Original default
            'cleanspecnet_dropout': net_cfg.get('cleanspecnet_dropout', 0.1),
        }

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
        
    def load_checkpoint(self, checkpoint_path):
        """Enhanced checkpoint loading with prefix handling"""
        print(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats and strip 'model.' prefix if present
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Strip 'model.' prefix from keys if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load with error handling
        try:
            self.model.load_state_dict(new_state_dict, strict=True)
            print(" Model loaded successfully with strict=True")
        except RuntimeError as e:
            print(f" Strict loading failed: {e}")
            print("Attempting to load with strict=False...")
            self.model.load_state_dict(new_state_dict, strict=False)
            print(" Model loaded with strict=False")
            
        print(f" Loaded checkpoint from {checkpoint_path}")

# Import torch at the end to avoid circular imports
import torch
