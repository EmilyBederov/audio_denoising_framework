#!/usr/bin/env python
# coding: utf-8

### Real-time inference script for original U-Net speech enhancement model ###
import os
import sys
import time
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from scipy import signal as sg
from pypesq import pesq as PESQ
from pystoi import stoi as STOI

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

### Original Full U-Net model definition in PyTorch ###
class UNetModel(nn.Module):
    def __init__(self):
        super(UNetModel, self).__init__()
        
        # Encoder
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
            nn.Conv2d(64, 128, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 2), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.enc6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.enc7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.enc8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder with padding calculated for each layer
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=(5, 5), stride=(1, 2), padding=(2, 2), output_padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        # Decoder with skip connections
        d1 = self.dec1(e8)
        if d1.size() != e7.size():
            d1 = F.interpolate(d1, size=e7.size()[2:], mode='nearest')
            
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        if d2.size() != e6.size():
            d2 = F.interpolate(d2, size=e6.size()[2:], mode='nearest')
            
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        if d3.size() != e5.size():
            d3 = F.interpolate(d3, size=e5.size()[2:], mode='nearest')
            
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        if d4.size() != e4.size():
            d4 = F.interpolate(d4, size=e4.size()[2:], mode='nearest')
            
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        if d5.size() != e3.size():
            d5 = F.interpolate(d5, size=e3.size()[2:], mode='nearest')
            
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        if d6.size() != e2.size():
            d6 = F.interpolate(d6, size=e2.size()[2:], mode='nearest')
            
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        if d7.size() != e1.size():
            d7 = F.interpolate(d7, size=e1.size()[2:], mode='nearest')
            
        d8 = self.dec8(torch.cat([d7, e1], dim=1))
        if d8.size() != x.size():
            d8 = F.interpolate(d8, size=x.size()[2:], mode='nearest')
        
        return d8

### Function for pre-processing ###
def pre_processing(data, Fs, down_sample):
    # Transform stereo into monoral
    if data.ndim == 2:
        wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
    else:
        wavdata = data
    
    # Downsample if necessary
    if down_sample is not None:
        wavdata = sg.resample_poly(wavdata, down_sample, Fs)
        Fs = down_sample
    
    return wavdata, Fs

### Function for reading an audio and getting the STFT ###
def read_evaldata(file_path, down_sample, frame_length, frame_shift, num_frames):
    # Initialize list
    x = []
    ang_x = []
    
    # Read .wav file and get pre-process
    wavdata, Fs = sf.read(file_path)
    wavdata, Fs = pre_processing(wavdata, Fs, down_sample)
    
    # Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    
    # Execute STFT
    _, _, dft = sg.stft(wavdata, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
    dft = dft[:-1].T # Remove the last point and get transpose
    ang = np.angle(dft) # Preserve the phase
    spec = np.log10(np.abs(dft) + 1e-10)  # Add small epsilon to avoid log(0)
    
    # Crop the temporal frames into input size
    num_seg = math.floor(spec.shape[0] / num_frames)
    for j in range(num_seg):
        # Add results to list sequentially
        x.append(spec[int(j*num_frames) : int((j+1)*num_frames), :])
        ang_x.append(ang[int(j*num_frames) : int((j+1)*num_frames), :])
    
    # Convert into numpy array
    x = np.array(x)
    ang_x = np.array(ang_x)
    
    return wavdata, Fs, x, ang_x

### Function for reconstructing a waveform ###
def reconstruct_wave(eval_y, ang_x, Fs, frame_length, frame_shift):
    # Construct the spectrogram by concatenating all segments
    Y = np.reshape(eval_y, (-1, eval_y.shape[-1]))
    ang = np.reshape(ang_x, (-1, ang_x.shape[-1]))
    
    # The Y and arg can be transpose for processing
    Y, ang = Y.T, ang.T
    
    # Restore the magnitude of STFT
    Y = np.power(10, Y)
    
    # Retrieve the phase from original wave
    Y = Y * np.exp(1j*ang)
    
    # Add the last frequency bin along with frequency axis
    Y = np.append(Y, Y[-1, :][np.newaxis,:], axis=0)
    
    # Get the inverse STFT
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    _, rec_wav = sg.istft(Y, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
    
    return rec_wav, Fs

### Function for measuring time per frame in milliseconds ###
def time_per_frame(model, input_frames, batch_size=1, num_runs=50):
    """Measure average processing time per frame in milliseconds"""
    model.eval()
    
    # Ensure input is on the correct device
    input_tensor = torch.tensor(input_frames, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Warmup runs
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_tensor[:batch_size])
    
    # Timed runs
    frame_times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_tensor[:batch_size])
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        frame_times.append(elapsed_ms / batch_size)  # Time per frame
    
    # Calculate statistics
    avg_time = sum(frame_times) / len(frame_times)
    min_time = min(frame_times)
    max_time = max(frame_times)
    
    return avg_time, min_time, max_time

### Main function for real-time inference ###
def run_inference(noisy_file):
    # Set parameters
    down_sample = 16000    # Downsampling rate (Hz)
    frame_length = 0.032   # STFT window width (second)
    frame_shift = 0.016    # STFT window shift (second)
    num_frames = 16        # The number of frames for an input
    
    # Create output directory
    output_dir = "./audio_data/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalization parameters (default values)
    min_x = -9.188868273733446
    max_x = -0.2012536819610933
    
    # Load model
    model = UNetModel().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total model parameters: {total_params:,}')
    
    # For the proof of concept, we're using the model without trained weights
    print("Using the original 8-layer U-Net model architecture")
    model.eval()
    
    # Read audio and compute STFT
    print(f"\nProcessing file: {os.path.basename(noisy_file)}")
    
    # Preprocessing time
    start_time_preprocess = time.time()
    mix_wav, Fs, eval_x, ang_x = read_evaldata(noisy_file, down_sample, frame_length, frame_shift, num_frames)
    eval_x = (eval_x - min_x) / (max_x - min_x)  # Normalize
    end_time_preprocess = time.time()
    preprocess_time = (end_time_preprocess - start_time_preprocess) * 1000  # milliseconds
    
    # For RTF calculation
    total_audio_duration = mix_wav.shape[0] / Fs
    total_audio_duration_ms = total_audio_duration * 1000  # milliseconds
    
    # Inference time
    start_time_inference = time.time()
    
    # Process in batches to avoid memory issues
    batch_size = 1  # Process one frame at a time for accurate timing
    num_batches = math.ceil(eval_x.shape[0] / batch_size)
    eval_y = np.zeros_like(eval_x)
    
    batch_times = []
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, eval_x.shape[0])
            
            # Convert batch to tensor
            batch_x = torch.tensor(eval_x[start_idx:end_idx], dtype=torch.float32).unsqueeze(1).to(device)
            
            # Time this batch
            batch_start = time.time()
            output = model(batch_x)
            batch_end = time.time()
            batch_time = (batch_end - batch_start) * 1000  # ms
            batch_times.append(batch_time)
            
            # Convert back to numpy
            eval_y[start_idx:end_idx] = output.squeeze(1).cpu().numpy()
    
    end_time_inference = time.time()
    inference_time = (end_time_inference - start_time_inference) * 1000  # milliseconds
    
    # Postprocessing time
    start_time_postprocess = time.time()
    
    # Restore the scale before normalization
    eval_y = eval_y * (max_x - min_x) + min_x
    
    # Reconstruct waveform
    sep_wav, Fs = reconstruct_wave(eval_y, ang_x, Fs, frame_length, frame_shift)
    
    end_time_postprocess = time.time()
    postprocess_time = (end_time_postprocess - start_time_postprocess) * 1000  # milliseconds
    
    # Calculate RTF
    rtf = inference_time / total_audio_duration_ms
    
    # Measure per-frame processing time
    if len(eval_x) > 0:
        print("Measuring per-frame processing time...")
        frame_avg_ms, frame_min_ms, frame_max_ms = time_per_frame(model, eval_x[0:1], batch_size=1, num_runs=50)
        
        # Calculate frames per second
        frames_per_second = 1000 / frame_avg_ms if frame_avg_ms > 0 else 0
    else:
        frame_avg_ms, frame_min_ms, frame_max_ms = 0, 0, 0
        frames_per_second = 0
    
    # Print results
    print("\n===== ORIGINAL U-NET MODEL (8 LAYERS) =====")
    print(f"Model parameters: {total_params:,}")
    print(f"Audio duration: {total_audio_duration:.2f} seconds ({total_audio_duration_ms:.1f} ms)")
    print(f"Preprocessing time: {preprocess_time:.1f} ms")
    print(f"Neural network inference time: {inference_time:.1f} ms")
    print(f"Postprocessing time: {postprocess_time:.1f} ms")
    print(f"Average processing time per frame: {frame_avg_ms:.2f} ms")
    print(f"Min/Max frame time: {frame_min_ms:.2f} ms / {frame_max_ms:.2f} ms")
    print(f"Frames processed per second: {frames_per_second:.1f}")
    print(f"Real-Time Factor (RTF): {rtf:.4f}")
    
    if rtf < 1.0:
        print(f"The model runs {1/rtf:.2f}x faster than real-time")
    else:
        print(f"The model runs {rtf:.2f}x slower than real-time")
    
    return rtf, frame_avg_ms

# Run as main script
if __name__ == "__main__":
    # Check if a file was provided
    if len(sys.argv) > 1:
        noisy_file = sys.argv[1]
        if not os.path.exists(noisy_file):
            print(f"Error: File {noisy_file} not found")
            sys.exit(1)
        run_inference(noisy_file)
    else:
        # Use first available file in evaluation directory
        eval_dir = "./audio_data/evaluation/NOISY"
        if not os.path.exists(eval_dir):
            print(f"Error: Evaluation directory {eval_dir} not found")
            sys.exit(1)
            
        files = sorted(glob.glob(f"{eval_dir}/*.wav"))
        if not files:
            print(f"Error: No WAV files found in {eval_dir}")
            sys.exit(1)
        
        # Run inference on first file
        run_inference(files[0])
