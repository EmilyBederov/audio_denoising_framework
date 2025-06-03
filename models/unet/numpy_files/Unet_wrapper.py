# models/unet_wrapper.py
from core.base_model import BaseModel
from models.unet.model import UNetModel

class UNetWrapper(BaseModel):
    """Wrapper for UNet model"""
    
    def _extract_model_params(self, config):
        """Extract UNet-specific parameters from config"""
        return {
            'input_channels': config.get('input_channels', 1),
            'output_channels': config.get('output_channels', 1),
            'base_channels': config.get('base_channels', 32),
            'depth': config.get('depth', 3),
        }
        
    def get_inference_input(self, noisy_audio, **kwargs):
        """Prepare input for inference by computing spectrogram"""
        import torch
        import numpy as np
        from scipy import signal as sg
        
        # Convert to numpy if it's a tensor
        if isinstance(noisy_audio, torch.Tensor):
            noisy_audio = noisy_audio.cpu().numpy()
            
        # Convert stereo to mono if needed
        if noisy_audio.ndim > 1 and noisy_audio.shape[1] > 1:
            noisy_audio = np.mean(noisy_audio, axis=1)
            
        # Get STFT parameters
        sample_rate = self.config.get('sample_rate', 16000)
        frame_length = self.config.get('frame_length', 0.032)
        frame_shift = self.config.get('frame_shift', 0.016)
        
        # Calculate window parameters
        FL = round(frame_length * sample_rate)
        FS = round(frame_shift * sample_rate)
        OL = FL - FS
        
        # Calculate STFT
        _, _, dft = sg.stft(noisy_audio, fs=sample_rate, window='hann', nperseg=FL, noverlap=OL)
        dft = dft[:-1].T  # Remove the last point and get transpose
        
        # Save phase for reconstruction (returned in kwargs)
        phase = np.angle(dft)
        
        # Convert to log magnitude
        magnitude = np.log10(np.abs(dft))
        
        # Normalize
        min_val = self.config.get('min_val', magnitude.min())
        max_val = self.config.get('max_val', magnitude.max())
        magnitude_norm = (magnitude - min_val) / (max_val - min_val)
        
        # Reshape into batches based on num_frames
        num_frames = self.config.get('num_frames', 16)
        num_segments = magnitude_norm.shape[0] // num_frames
        segments = []
        
        for i in range(num_segments):
            segment = magnitude_norm[i*num_frames:(i+1)*num_frames, :]
            segments.append(segment)
            
        # Convert to tensor
        segments = np.array(segments)
        input_tensor = torch.tensor(segments, dtype=torch.float32).unsqueeze(1).to(noisy_audio.device)
        
        # Store phase for reconstruction
        kwargs['phase'] = phase
        kwargs['original_shape'] = magnitude.shape
        kwargs['min_val'] = min_val
        kwargs['max_val'] = max_val
        
        return input_tensor