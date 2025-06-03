#!/usr/bin/env python3
"""
Setup script to integrate UNet into your existing framework
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the necessary directory structure for UNet integration"""
    
    # Create UNet model directories
    unet_dirs = [
        "models/unet",
        "models/unet/models", 
        "configs/configs-unet"
    ]
    
    for dir_path in unet_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def create_unet_model_file():
    """Create the UNet model file"""
    
    unet_model_content = '''# models/unet/models/unet.py
"""
UNet model implementation for your framework
This matches the structure expected by your model factory
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    UNet model for audio denoising using spectrograms
    """
    
    def __init__(self, input_channels=1, output_channels=1, base_channels=32, depth=3):
        super(UNet, self).__init__()
        
        # Store configuration
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth
        
        # Build encoder
        self.encoder = nn.ModuleList()
        in_channels = input_channels
        
        for i in range(depth):
            out_channels = base_channels * (2 ** i)
            
            if i == 0:
                # First encoder layer
                layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(5, 7), 
                             stride=(1, 2), padding=(2, 3)),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                )
            elif i == depth - 1:
                # Last encoder layer
                layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), 
                             stride=(2, 2), padding=(2, 2)),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                )
            else:
                # Middle encoder layers
                layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(5, 7), 
                             stride=(1, 2), padding=(2, 3)),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                )
            
            self.encoder.append(layer)
            in_channels = out_channels
        
        # Build decoder
        self.decoder = nn.ModuleList()
        
        for i in range(depth - 1, 0, -1):
            in_channels = base_channels * (2 ** i)
            out_channels = base_channels * (2 ** (i - 1))
            
            if i == depth - 1:
                # First decoder layer (no skip connection)
                layer = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(5, 5), 
                                     stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            else:
                # Other decoder layers (with skip connections)
                layer = nn.Sequential(
                    nn.ConvTranspose2d(in_channels * 2, out_channels, kernel_size=(5, 7), 
                                     stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            
            self.decoder.append(layer)
        
        # Final output layer
        final_in_channels = base_channels * 2  # Skip connection from first encoder
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(final_in_channels, output_channels, kernel_size=(5, 7), 
                             stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder forward pass
        for enc_layer in self.encoder:
            x = enc_layer(x)
            encoder_outputs.append(x)
        
        # Decoder forward pass with skip connections
        skip_connections = encoder_outputs[:-1][::-1]  # Reverse order, exclude last
        
        for i, dec_layer in enumerate(self.decoder):
            if i == 0:
                # First decoder layer (no skip connection)
                x = dec_layer(x)
                # Resize to match skip connection if needed
                if len(skip_connections) > 0:
                    target_size = skip_connections[i].size()[2:]
                    x = F.interpolate(x, size=target_size, mode='nearest')
            else:
                # Concatenate with skip connection
                skip_idx = i - 1
                if skip_idx < len(skip_connections):
                    skip = skip_connections[skip_idx]
                    # Ensure sizes match
                    if x.size()[2:] != skip.size()[2:]:
                        x = F.interpolate(x, size=skip.size()[2:], mode='nearest')
                    x = torch.cat([x, skip], dim=1)
                
                x = dec_layer(x)
                
                # Resize for next iteration if needed
                next_skip_idx = i
                if next_skip_idx < len(skip_connections):
                    target_size = skip_connections[next_skip_idx].size()[2:]
                    x = F.interpolate(x, size=target_size, mode='nearest')
        
        # Final layer with first encoder output (skip connection)
        if len(encoder_outputs) > 0:
            first_encoder_output = encoder_outputs[0]
            if x.size()[2:] != first_encoder_output.size()[2:]:
                x = F.interpolate(x, size=first_encoder_output.size()[2:], mode='nearest')
            x = torch.cat([x, first_encoder_output], dim=1)
        
        x = self.final_layer(x)
        
        return x
'''
    
    with open("models/unet/models/unet.py", "w") as f:
        f.write(unet_model_content)
    print("✓ Created UNet model file")

def create_unet_wrapper():
    """Create the UNet wrapper file"""
    
    wrapper_content = '''# models/unet/unet_wrapper.py
from core.base_model import BaseModel

class UNetWrapper(BaseModel):
    """UNet wrapper that fits your existing framework structure"""
    
    def __init__(self, model_class, config):
        """
        Initialize UNet wrapper to match your BaseModel structure
        
        Args:
            model_class: The UNet model class from models/unet/models/unet.py
            config: Configuration dictionary
        """
        # Extract network config for UNet
        network_config = {
            'input_channels': config.get('input_channels', 1),
            'output_channels': config.get('output_channels', 1), 
            'base_channels': config.get('base_channels', 32),
            'depth': config.get('depth', 3)
        }
        
        # Update config with network_config
        config['network_config'] = network_config
        
        # Call parent constructor
        super().__init__(model_class, config)
        
        # Store UNet-specific parameters
        self.sample_rate = config.get('sample_rate', 16000)
        self.frame_length = config.get('frame_length', 0.032)
        self.frame_shift = config.get('frame_shift', 0.016)
        self.num_frames = config.get('num_frames', 16)
        self.min_val = config.get('min_val', -9.188868273733446)
        self.max_val = config.get('max_val', -0.2012536819610933)
        
    def preprocess_for_inference(self, audio_path):
        """
        Preprocess audio file for UNet inference
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            tuple: (preprocessed_tensor, phase_info) for reconstruction
        """
        import numpy as np
        import soundfile as sf
        from scipy import signal as sg
        import torch
        import math
        
        # Read audio
        audio, sample_rate = sf.read(audio_path)
        
        # Convert to mono
        if audio.ndim == 2:
            audio = 0.5 * audio[:, 0] + 0.5 * audio[:, 1]
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            audio = sg.resample_poly(audio, self.sample_rate, sample_rate)
        
        # Calculate STFT parameters
        FL = round(self.frame_length * self.sample_rate)
        FS = round(self.frame_shift * self.sample_rate)
        OL = FL - FS
        
        # Compute STFT
        _, _, dft = sg.stft(audio, fs=self.sample_rate, window='hann', nperseg=FL, noverlap=OL)
        dft = dft[:-1].T  # Remove last point and transpose
        
        # Store phase for reconstruction
        phase = np.angle(dft)
        
        # Convert to log magnitude
        log_magnitude = np.log10(np.abs(dft) + 1e-10)
        
        # Normalize
        log_magnitude = (log_magnitude - self.min_val) / (self.max_val - self.min_val)
        
        # Crop into segments
        num_segments = math.floor(log_magnitude.shape[0] / self.num_frames)
        segments = []
        
        for i in range(num_segments):
            start_idx = i * self.num_frames
            end_idx = (i + 1) * self.num_frames
            segment = log_magnitude[start_idx:end_idx, :]
            segments.append(segment)
        
        if segments:
            # Convert to tensor with batch and channel dimensions
            segments = np.array(segments)
            input_tensor = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)  # Add channel dim
            
            # Store reconstruction info
            reconstruction_info = {
                'phase': phase,
                'original_audio': audio,
                'sample_rate': self.sample_rate,
                'num_segments': num_segments
            }
            
            return input_tensor, reconstruction_info
        else:
            # Return empty tensor if no segments
            return torch.zeros((1, 1, self.num_frames, 513)), None
    
    def postprocess_output(self, model_output, reconstruction_info):
        """
        Convert model output back to audio
        
        Args:
            model_output: Model output tensor
            reconstruction_info: Info needed for reconstruction
            
        Returns:
            numpy array: Reconstructed audio
        """
        import numpy as np
        from scipy import signal as sg
        
        if reconstruction_info is None:
            return np.zeros(16000)  # Return 1 second of silence
        
        # Convert to numpy and remove batch/channel dimensions
        output = model_output.squeeze().cpu().numpy()
        
        # Denormalize
        output = output * (self.max_val - self.min_val) + self.min_val
        
        # Reshape and reconstruct spectrogram
        if output.ndim == 3:
            # Multiple segments
            output = output.reshape(-1, output.shape[-1])
        
        # Get phase info
        phase = reconstruction_info['phase']
        
        # Truncate to match phase length
        min_len = min(output.shape[0], phase.shape[0])
        output = output[:min_len]
        phase = phase[:min_len]
        
        # Convert back to complex spectrogram
        magnitude = np.power(10, output)
        complex_spec = magnitude * np.exp(1j * phase)
        
        # Add frequency bin and transpose
        complex_spec = np.vstack([complex_spec.T, complex_spec.T[-1][np.newaxis, :]])
        
        # Inverse STFT
        FL = round(self.frame_length * self.sample_rate)
        FS = round(self.frame_shift * self.sample_rate)
        OL = FL - FS
        
        _, reconstructed_audio = sg.istft(complex_spec, fs=self.sample_rate, 
                                         window='hann', nperseg=FL, noverlap=OL)
        
        return reconstructed_audio
'''
    
    with open("models/unet/unet_wrapper.py", "w") as f:
        f.write(wrapper_content)
    print("✓ Created UNet wrapper file")

def create_unet_config():
    """Create the UNet config file"""
    
    config_content = '''# configs/configs-unet/unet-config.yaml
# UNet configuration that matches your framework structure

# Model parameters
model_name: unet
input_channels: 1
output_channels: 1
base_channels: 32
depth: 3

# Audio processing parameters (UNet-specific)
sample_rate: 16000
frame_length: 0.032  # STFT window width in seconds
frame_shift: 0.016   # STFT window shift in seconds
num_frames: 16       # Number of frames per input segment

# Normalization parameters (computed from training data)
min_val: -9.188868273733446
max_val: -0.2012536819610933

# Training parameters
batch_size: 16
epochs: 30
learning_rate: 0.0001
weight_decay: 0.0001

# Data paths (modify these to match your data structure)
trainset:
  csv_path: "data/audio_data/training_pairs_16khz.csv"

valset:
  csv_path: "data/audio_data/evaluation_pairs_16khz.csv"

# Optional test set
testset:
  csv_path: "data/audio_data/evaluation_pairs_16khz.csv"

# Data processing parameters that match your existing data loader
n_fft: 1024
hop_length: 256
win_length: 1024
power: 1.0
max_length: 65536  # Maximum audio length

# Worker settings
num_workers: 4

# Checkpointing
save_path: "outputs/unet"

# Loss function
loss_type: "l1"  # Can be 'l1', 'mse', or 'huber'
'''
    
    with open("configs/configs-unet/unet-config.yaml", "w") as f:
        f.write(config_content)
    print("✓ Created UNet config file")

def update_model_factory():
    """Update the model factory to include UNet"""
    
    # Read the current model factory
    try:
        with open("core/model_factory.py", "r") as f:
            content = f.read()
        
        # Check if UNet is already added
        if "'unet':" in content and "UNetWrapper" in content:
            print("✓ UNet already in model factory")
            return
        
        # Add UNet to MODEL_MAPPING
        if "MODEL_MAPPING = {" in content:
            # Find the MODEL_MAPPING section and add UNet
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                new_lines.append(line)
                if "'cleanunet2':" in line and "CleanUNet2Wrapper" in line:
                    # Add UNet after CleanUNet2
                    indent = len(line) - len(line.lstrip())
                    unet_line = ' ' * indent + "'unet': 'models.unet.unet_wrapper.UNetWrapper',"
                    new_lines.append(unet_line)
            
            # Add UNet import to _get_model_class
            for i, line in enumerate(new_lines):
                if "elif model_name == 'unet':" not in content and "def _get_model_class" in line:
                    # Find the end of the function and add UNet case
                    j = i
                    while j < len(new_lines) and not (new_lines[j].strip().startswith("elif model_name == 'ftcrngan'") or 
                                                     new_lines[j].strip().startswith("else:")):
                        j += 1
                    
                    # Insert UNet case before else clause
                    if j < len(new_lines):
                        indent = "        "  # Match existing indentation
                        unet_case = [
                            f"{indent}elif model_name == 'unet':",
                            f"{indent}    # New UNet import", 
                            f"{indent}    from models.unet.models.unet import UNet",
                            f"{indent}    return UNet"
                        ]
                        new_lines[j:j] = unet_case
                    break
            
            # Write back the updated content
            with open("core/model_factory.py", "w") as f:
                f.write('\n'.join(new_lines))
            
            print("✓ Updated model factory with UNet support")
        else:
            print("⚠ Could not automatically update model factory. Please add UNet manually.")
            
    except Exception as e:
        print(f"⚠ Error updating model factory: {e}")
        print("Please manually add UNet to your model factory")

def create_init_files():
    """Create __init__.py files for proper imports"""
    
    init_files = [
        "models/unet/__init__.py",
        "models/unet/models/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ Created {init_file}")

def test_integration():
    """Test that the integration works"""
    
    print("\n" + "="*50)
    print("Testing UNet Integration...")
    print("="*50)
    
    try:
        # Test model factory
        from core.model_factory import ModelFactory
        
        # Test config loading
        config = ModelFactory.load_model_config('unet')
        print("✓ Config loading works")
        
        # Test model creation
        model = ModelFactory.create_model('unet', config)
        print("✓ Model creation works")
        
        # Test forward pass
        import torch
        dummy_input = torch.randn(1, 1, 16, 513)  # Batch, channel, time, freq
        output = model(dummy_input)
        print(f"✓ Forward pass works - Input: {dummy_input.shape}, Output: {output.shape}")
        
        print("\n🎉 UNet integration successful!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print("Please check the error and try again.")

def main():
    """Main setup function"""
    
    print("UNet Integration Setup")
    print("=" * 30)
    
    # Step 1: Create directory structure
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Create model files
    print("\n2. Creating UNet model files...")
    create_unet_model_file()
    create_unet_wrapper()
    
    # Step 3: Create config
    print("\n3. Creating configuration...")
    create_unet_config()
    
    # Step 4: Update model factory
    print("\n4. Updating model factory...")
    update_model_factory()
    
    # Step 5: Create __init__ files
    print("\n5. Creating __init__ files...")
    create_init_files()
    
    # Step 6: Test integration
    print("\n6. Testing integration...")
    test_integration()
    
    print("\n" + "="*50)
    print("Setup Complete!")
    print("="*50)
    print("\nTo use UNet:")
    print("1. python run.py --config configs/configs-unet/unet-config.yaml --model unet --mode train")
    print("2. python run.py --config configs/configs-unet/unet-config.yaml --model unet --mode inference --input input.wav")

if __name__ == "__main__":
    main()