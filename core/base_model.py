# core/base_model.py
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

class BaseModel(nn.Module):
    """Base wrapper for all audio denoising models in the framework"""
    
    def __init__(self, model_class, config: Dict[str, Any]):
        """
        Initialize a model wrapper with configuration.
        
        Args:
            model_class: The actual model class to instantiate
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.model_name = config.get('model_name', 'unknown_model')
        
        # Extract network config
        self.network_config = config.get('network_config', {})
        
        # Initialize the model
        self.model = self._create_model(model_class)
        
    def _create_model(self, model_class):
        """Create the actual model instance using network_config"""
        return model_class(**self.network_config)
    
    def forward(self, *args, **kwargs):
        """Forward pass to the underlying model"""
        return self.model(*args, **kwargs)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        print(f"Loaded checkpoint from {checkpoint_path}")
        
    def save_checkpoint(self, checkpoint_path, optimizer=None, epoch=None, loss=None):
        """Save model weights to checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'config': self.config,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
