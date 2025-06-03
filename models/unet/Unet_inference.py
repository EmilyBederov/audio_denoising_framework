# models/unet/inference.py
import torch
import torch.nn.functional as F
import numpy as np
import time
import soundfile as sf
from scipy import signal as sg

def preprocess_audio(audio, sample_rate, target_sample_rate, frame_length, frame_shift):
    """Preprocess audio for UNet inference"""
    # Convert to mono if needed
    if audio.ndim > 1 and audio.shape[1] > 1:
        audio = 0.5 * audio[:, 0] + 0.5 * audio[:, 1]
    
    # Downsample if needed
    if sample_rate != target_sample_rate:
        audio = sg.resample_poly(audio, target_sample_rate, sample_rate)
    
    # Calculate STFT parameters
    FL = round(frame_length * target_sample_rate)
    FS = round(frame_shift * target_sample_rate)
    OL = FL - FS
    
    # Calculate STFT
    _, _, dft = sg.stft(audio, fs=target_sample_rate, window='hann', nperseg=FL, noverlap=OL)
    dft = dft[:-1].T  # Remove the last point and get transpose
    
    # Save phase for reconstruction
    phase = np.angle(dft)
    
    # Convert to log magnitude
    magnitude = np.log10(np.abs(dft))
    
    return magnitude, phase, target_sample_rate

def run_inference(model, audio_path, config, device='cuda'):
    """Run inference on an audio file with UNet model"""
    # Load model parameters
    target_sample_rate = config.get('sample_rate', 16000)
    frame_length = config.get('frame_length', 0.032)
    frame_shift = config.get('frame_shift', 0.016)
    num_frames = config.get('num_frames', 16)
    
    # Load audio
    audio, sample_rate = sf.read(audio_path)
    
    # Preprocess audio
    magnitude, phase, processed_sample_rate = preprocess_audio(
        audio, sample_rate, target_sample_rate, frame_length, frame_shift
    )
    
    # Crop into segments
    num_segments = magnitude.shape[0] // num_frames
    segments = []
    for i in range(num_segments):
        segment = magnitude[i*num_frames:(i+1)*num_frames, :]
        segments.append(segment)
    
    # Convert to tensor
    segments = np.array(segments)
    
    # Normalize
    if 'min_val' in config and 'max_val' in config:
        min_val, max_val = config['min_val'], config['max_val']
    else:
        min_val, max_val = segments.min(), segments.max()
        
    segments_normalized = (segments - min_val) / (max_val - min_val)
    
    # Convert to torch tensor
    input_tensor = torch.tensor(segments_normalized, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Start timing
    start_time = time.time()
    
    # Run inference in batches
    batch_size = 32
    output_segments = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i:i+batch_size]
            output = model(batch)
            output_segments.append(output.cpu().numpy())
    
    # End timing
    inference_time = time.time() - start_time
    
    # Concatenate all batches
    output_segments = np.vstack(output_segments)
    
    # Denormalize
    output_segments = output_segments * (max_val - min_val) + min_val
    
    # Reshape to original size
    output_segments = output_segments.reshape(-1, segments.shape[-1])
    
    # Truncate to match original size
    if output_segments.shape[0] > magnitude.shape[0]:
        output_segments = output_segments[:magnitude.shape[0], :]
    
    # Reconstruct audio
    FL = round(frame_length * processed_sample_rate)
    FS = round(frame_shift * processed_sample_rate)
    OL = FL - FS
    
    # Convert back to complex spectrogram
    output_segments = np.power(10, output_segments)
    complex_spec = output_segments * np.exp(1j * phase[:output_segments.shape[0], :])
    
    # Add the last frequency bin
    complex_spec = np.vstack([complex_spec.T, complex_spec.T[-1][np.newaxis, :]])
    
    # Inverse STFT
    _, reconstructed_audio = sg.istft(complex_spec, fs=processed_sample_rate, window='hann', nperseg=FL, noverlap=OL)
    
    # Calculate RTF
    audio_duration = len(audio) / sample_rate
    rtf = inference_time / audio_duration
    
    return reconstructed_audio, processed_sample_rate, rtf
