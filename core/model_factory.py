# core/model_factory.py
from typing import Dict, Any, Optional
import importlib
import yaml
from pathlib import Path

class ModelFactory:
    """Factory for creating model instances"""
    
    MODEL_MAPPING = {
        'cleanunet2': 'models.cleanunet2.cleanunet2_wrapper.CleanUNet2Wrapper',
        'cleanunet2_original': 'models.cleanunet2.cleanunet2_wrapper.CleanUNet2Wrapper',  # Use original implementation
        'unet': 'models.unet.unet_wrapper.UNetWrapper',
        'ftcrngan': 'models.ftcrngan.ftcrngan_wrapper.FTCRNGANWrapper',
    }
    
    @classmethod
    def create_model(cls, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Create a model instance by name with the given config.
        
        Args:
            model_name: Name of the model to create
            config: Model configuration dict (if None, loads from config file)
            
        Returns:
            Instantiated model wrapper
        """
        if model_name not in cls.MODEL_MAPPING:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Load config if not provided
        if config is None:
            config = cls.load_model_config(model_name)
            
        # Add model_name to config
        config['model_name'] = model_name
        
        # Import and instantiate the model wrapper
        module_path, class_name = cls.MODEL_MAPPING[model_name].rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_wrapper_class = getattr(module, class_name)
        
        # Get the model class (for cleanunet2, we use the original implementation)
        model_class = cls._get_model_class(model_name)
        
        # Create and return the model wrapper
        return model_wrapper_class(model_class, config)
    
    @staticmethod
    def _get_model_class(model_name: str):
        """Get the actual model class for the given model name"""
        if model_name in ['cleanunet2', 'cleanunet2_original']:
            # Use the ORIGINAL CleanUNet2 implementation
            from models.cleanunet2.models.cleanunet2 import CleanUNet2
            return CleanUNet2
        elif model_name == 'unet':
            from models.unet.models.unet import UNet
            return UNet
        elif model_name == 'ftcrngan':
            from models.ftcrngan.models.ftcrngan import FTCRNGAN
            return FTCRNGAN
        else:
            raise ValueError(f"Unknown model class for: {model_name}")
        
    @staticmethod
    def load_model_config(model_name: str):
        """Load model configuration from YAML file"""
        # Try different config naming patterns
        config_paths = [
            Path(f'configs/configs-{model_name}/{model_name}-original-config.yaml'),
            Path(f'configs/configs-{model_name}/{model_name}-config.yaml'),
            Path(f'configs/{model_name}.yaml'),
            Path(f'configs/{model_name}-original.yaml')
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        
        raise FileNotFoundError(f"Config not found for {model_name}. Tried: {[str(p) for p in config_paths]}")
