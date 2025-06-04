# models/unet/unet_wrapper.py
"""
UNet wrapper that integrates with the existing framework
Key difference: UNet processes spectrograms only, not audio + spectrogram like CleanUNet2
"""
import torch
import numpy as np
import soundfile as sf
from scipy import signal as sg
import math
import os
from core.base_model import BaseModel

class UNetWrapper(BaseModel):
    """UNet wrapper that handles spectrogram-only processing"""
    
    def __init__(self, model_class, config):
        """
        Initialize UNet wrapper to work with existing framework
        
        Args:
            model_class: The UNet model class
            config: Configuration dictionary
        """
        # Initialize the base model
        super().__init__(model_class, config)
        
        # Store UNet-specific parameters from config
        self.model_type = "spectrogram_only"  # Flag to help training manager
        
        # Audio processing parameters
        self.sample_rate = config.get('sample_rate', 16000)
        self.frame_length = config.get('frame_length', 0.032)
        self.frame_shift = config.get('frame_shift', 0.016)
        self.num_frames = config.get('num_frames', 16)
        
        # Normalization parameters (from original UNet training)
        self.min_val = config.get('min_val', -9.188868273733446)
        self.max_val = config.get('max_val', -0.2012536819610933)
        
        print(f"✅ UNet wrapper initialized - processes spectrograms only")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def forward(self, *args, **kwargs):
        """
        Forward pass that handles different input formats
        
        For UNet:
        - If given (audio, spectrogram) -> use spectrogram only
        - If given spectrogram only -> use directly
        """
        if len(args) == 2:
            # CleanUNet2 format: (audio, spectrogram) -> use only spectrogram for UNet
            _, spectrogram = args
            return self.model(spectrogram)
        elif len(args) == 1:
            # Direct spectrogram input
            spectrogram = args[0]
            return self.model(spectrogram)
        else:
            raise ValueError(f"UNet expects 1 or 2 inputs, got {len(args)}")
    
    def get_training_input_format(self):
        """Return the input format this model expects for training"""
        return "spectrogram_only"
    
    def prepare_training_batch(self, batch):
        """
        Prepare batch for UNet training
        
        UNet uses spectrogram -> spectrogram mapping
        Input: noisy spectrogram
        Target: clean spectrogram
        
        Args:
            batch: (clean_audio, clean_spec, noisy_audio, noisy_spec)
            
        Returns:
            inputs, targets for UNet training
        """
        if len(batch) == 4:
            clean_audio, clean_spec, noisy_audio, noisy_spec = batch
            
            # For UNet: input = noisy_spec, target = clean_spec
            inputs = noisy_spec  # [B, F, T]
            targets = clean_spec  # [B, F, T]
            
            # Add channel dimension if needed: [B, F, T] -> [B, 1, F, T]
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
                
            # Transpose to [B, 1, T, F] format expected by UNet
            inputs = inputs.transpose(-2, -1)  # [B, 1, F, T] -> [B, 1, T, F]
            targets = targets.transpose(-2, -1)  # [B, 1, F, T] -> [B, 1, T, F]
            
            return inputs, targets
        else:
            raise ValueError(f"Expected batch with 4 elements, got {len(batch)}")
    
    def preprocess_for_inference(self, audio_path):
        """
        Preprocess audio file for UNet inference
        Uses the exact same preprocessing as the original UNet training
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            tuple: (preprocessed_tensor, reconstruction_info)
        """
        # Read and preprocess audio - EXACT from original
        audio, sample_rate = sf.read(audio_path)
        
        # Convert stereo to mono
        if audio.ndim == 2:
            audio = 0.5 * audio[:, 0] + 0.5 * audio[:, 1]
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            audio = sg.resample_poly(audio, self.sample_rate, sample_rate)
        
        # Calculate STFT parameters - EXACT from original
        FL = round(self.frame_length * self.sample_rate)
        FS = round(self.frame_shift * self.sample_rate)
        OL = FL - FS
        
        # Compute STFT - EXACT from original
        _, _, dft = sg.stft(audio, fs=self.sample_rate, window='hann', nperseg=FL, noverlap=OL)
        dft = dft[:-1].T  # Remove last point and transpose
        
        # Store phase for reconstruction
        phase = np.angle(dft)
        
        # Convert to log magnitude - EXACT from original
        log_magnitude = np.log10(np.abs(dft))
        
        # Normalize - EXACT from original
        log_magnitude_norm = (log_magnitude - self.min_val) / (self.max_val - self.min_val)
        
        # Crop into segments - EXACT from original
        num_segments = math.floor(log_magnitude_norm.shape[0] / self.num_frames)
        segments = []
        
        for i in range(num_segments):
            start_idx = i * self.num_frames
            end_idx = (i + 1) * self.num_frames
            segment = log_magnitude_norm[start_idx:end_idx, :]
            segments.append(segment)
        
        if segments:
            # Convert to tensor: [N, T, F] -> [N, 1, T, F]
            segments = np.array(segments)
            input_tensor = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)
            
            # Store reconstruction info
            reconstruction_info = {
                'phase': phase,
                'original_audio': audio,
                'sample_rate': self.sample_rate,
                'num_segments': num_segments,
                'original_shape': log_magnitude.shape
            }
            
            return input_tensor, reconstruction_info
        else:
            # Return dummy data if no segments
            dummy_tensor = torch.zeros((1, 1, self.num_frames, 513))
            return dummy_tensor, None
    
    def postprocess_output(self, model_output, reconstruction_info):
        """
        Convert model output back to audio
        Uses the exact same postprocessing as the original UNet
        
        Args:
            model_output: Model output tensor [N, 1, T, F]
            reconstruction_info: Info needed for reconstruction
            
        Returns:
            numpy array: Reconstructed audio
        """
        if reconstruction_info is None:
            return np.zeros(16000)  # Return 1 second of silence
        
        # Convert to numpy and remove batch/channel dimensions
        output = model_output.squeeze().cpu().numpy()  # [N, T, F] or [T, F]
        
        # Handle single segment case
        if output.ndim == 2:
            output = output[np.newaxis, ...]  # [1, T, F]
        
        # Denormalize - EXACT from original
        output = output * (self.max_val - self.min_val) + self.min_val
        
        # Reshape to full spectrogram - EXACT from original
        output_reshaped = output.reshape(-1, output.shape[-1])  # [N*T, F]
        
        # Get phase and truncate to match
        phase = reconstruction_info['phase']
        min_len = min(output_reshaped.shape[0], phase.shape[0])
        output_reshaped = output_reshaped[:min_len]
        phase = phase[:min_len]
        
        # Convert back to complex spectrogram - EXACT from original
        magnitude = np.power(10, output_reshaped)
        complex_spec = magnitude * np.exp(1j * phase)
        
        # Transpose and add frequency bin - EXACT from original
        complex_spec = complex_spec.T  # [F, T]
        complex_spec = np.vstack([complex_spec, complex_spec[-1][np.newaxis, :]])  # Add last freq bin
        
        # Inverse STFT - EXACT from original
        FL = round(self.frame_length * self.sample_rate)
        FS = round(self.frame_shift * self.sample_rate)
        OL = FL - FS
        
        _, reconstructed_audio = sg.istft(complex_spec, fs=self.sample_rate, 
                                         window='hann', nperseg=FL, noverlap=OL)
        
        return reconstructed_audio
    
    def inference(self, audio_path, output_path=None):
        """
        Complete inference pipeline
        
        Args:
            audio_path: Input audio file path
            output_path: Output audio file path (optional)
            
        Returns:
            Reconstructed audio array
        """
        # Preprocess
        input_tensor, reconstruction_info = self.preprocess_for_inference(audio_path)
        
        # Move to device
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        reconstructed_audio = self.postprocess_output(output, reconstruction_info)
        
        # Save if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, reconstructed_audio, self.sample_rate)
            print(f"✅ Saved enhanced audio to: {output_path}")
        
        return reconstructed_audio
