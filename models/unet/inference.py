# #!/usr/bin/env python
# # coding: utf-8

# ### Inference script for U-Net speech enhancement ###
# import os
# import sys
# import time
# import glob
# import math
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import soundfile as sf
# from scipy import signal as sg
# from pypesq import pesq as PESQ
# from pystoi import stoi as STOI

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# ### U-Net model definition in PyTorch ###
# class UNetModel(nn.Module):
#     def __init__(self):
#         super(UNetModel, self).__init__()
        
#         # Simplified encoder (just 3 layers)
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3)),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.enc3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2)
#         )
        
#         # Simplified decoder (just 3 layers)
#         self.dec1 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
        
#         self.dec2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 32, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )
        
#         self.dec3 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         # Encoder
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
        
#         # Decoder with skip connections
#         d1 = self.dec1(e3)
#         d1 = F.interpolate(d1, size=e2.size()[2:], mode='nearest')
#         d2 = self.dec2(torch.cat([d1, e2], dim=1))
#         d2 = F.interpolate(d2, size=e1.size()[2:], mode='nearest')
#         d3 = self.dec3(torch.cat([d2, e1], dim=1))
#         d3 = F.interpolate(d3, size=x.size()[2:], mode='nearest')
        
#         return d3

# ### Function for pre-processing ###
# def pre_processing(data, Fs, down_sample):
    
#     #Transform stereo into monoral
#     if data.ndim == 2:
#         wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
#     else:
#         wavdata = data
    
#     #Downsample if necessary
#     if down_sample is not None:
#         wavdata = sg.resample_poly(wavdata, down_sample, Fs)
#         Fs = down_sample
    
#     return wavdata, Fs

# ### Function for reading an audio and getting the STFT ###
# def read_evaldata(file_path, down_sample, frame_length, frame_shift, num_frames):
    
#     #Initialize list
#     x = []
#     ang_x = []
    
#     #Read .wav file and get pre-process
#     wavdata, Fs = sf.read(file_path)
#     wavdata, Fs = pre_processing(wavdata, Fs, down_sample)
    
#     #Calculate the index of window size and overlap
#     FL = round(frame_length * Fs)
#     FS = round(frame_shift * Fs)
#     OL = FL - FS
    
#     #Execute STFT
#     _, _, dft = sg.stft(wavdata, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
#     dft = dft[:-1].T #Remove the last point and get transpose
#     ang = np.angle(dft) #Preserve the phase
#     spec = np.log10(np.abs(dft) + 1e-10)  # Add small epsilon to avoid log(0)
    
#     #Crop the temporal frames into input size
#     num_seg = math.floor(spec.shape[0] / num_frames)
#     for j in range(num_seg):
#         #Add results to list sequentially
#         x.append(spec[int(j*num_frames) : int((j+1)*num_frames), :])
#         ang_x.append(ang[int(j*num_frames) : int((j+1)*num_frames), :])
    
#     #Convert into numpy array
#     x = np.array(x)
#     ang_x = np.array(ang_x)
    
#     return wavdata, Fs, x, ang_x

# ### Function for reconstructing a waveform ###
# def reconstruct_wave(eval_y, ang_x, Fs, frame_length, frame_shift):
    
#     #Construct the spectrogram by concatenating all segments
#     Y = np.reshape(eval_y, (-1, eval_y.shape[-1]))
#     ang = np.reshape(ang_x, (-1, ang_x.shape[-1]))
    
#     #The Y and arg can be transpose for processing
#     Y, ang = Y.T, ang.T
    
#     #Restore the magnitude of STFT
#     Y = np.power(10, Y)
    
#     #Retrieve the phase from original wave
#     Y = Y * np.exp(1j*ang)
    
#     #Add the last frequency bin along with frequency axis
#     Y = np.append(Y, Y[-1, :][np.newaxis,:], axis=0)
    
#     #Get the inverse STFT
#     FL = round(frame_length * Fs)
#     FS = round(frame_shift * Fs)
#     OL = FL - FS
#     _, rec_wav = sg.istft(Y, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
    
#     return rec_wav, Fs

# ### Function for real-time inference with RTF calculation ###
# def real_time_inference(noisy_file, output_dir="./audio_data/output", model_path="./models/unet_model.pth"):
#     # Set parameters
#     down_sample = 16000    # Downsampling rate (Hz)
#     frame_length = 0.032   # STFT window width (second)
#     frame_shift = 0.016    # STFT window shift (second)
#     num_frames = 16        # The number of frames for an input
    
#     # Normalization parameters (loaded or default)
#     try:
#         norm_values = np.load('./numpy_files/norm_values.npy', allow_pickle=True).item()
#         min_x = norm_values['min']
#         max_x = norm_values['max']
#         print(f"Loaded normalization values: min={min_x}, max={max_x}")
#     except:
#         # Use default values
#         min_x = -9.188868273733446
#         max_x = -0.2012536819610933
#         print(f"Using default normalization values: min={min_x}, max={max_x}")
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load model
#     model = UNetModel().to(device)
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f'Total model parameters: {total_params:,}')
    
#     # Try to load model weights
#     try:
#         model.load_state_dict(torch.load(model_path))
#         print(f"Loaded model from '{model_path}'")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         # Try to find checkpoints
#         checkpoints = sorted(glob.glob('./models/unet_checkpoint_*.pth'))
#         if checkpoints:
#             latest_checkpoint = checkpoints[-1]
#             print(f"Loading latest checkpoint: {latest_checkpoint}")
#             model.load_state_dict(torch.load(latest_checkpoint))
#         else:
#             print("No model or checkpoint found. Using untrained model for proof of concept...")
    
#     model.eval()
    
#     # Read audio and compute STFT
#     start_time_total = time.time()
    
#     # Preprocessing time
#     start_time_preprocess = time.time()
#     mix_wav, Fs, eval_x, ang_x = read_evaldata(noisy_file, down_sample, frame_length, frame_shift, num_frames)
#     eval_x = (eval_x - min_x) / (max_x - min_x)  # Normalize
#     end_time_preprocess = time.time()
#     preprocess_time = end_time_preprocess - start_time_preprocess
    
#     # For RTF calculation
#     total_audio_duration = mix_wav.shape[0] / Fs
    
#     # Inference time
#     start_time_inference = time.time()
    
#     # Process in batches to avoid memory issues
#     batch_size = 32
#     num_batches = math.ceil(eval_x.shape[0] / batch_size)
#     eval_y = np.zeros_like(eval_x)
    
#     with torch.no_grad():
#         for i in range(num_batches):
#             start_idx = i * batch_size
#             end_idx = min((i + 1) * batch_size, eval_x.shape[0])
            
#             # Convert batch to tensor
#             batch_x = torch.tensor(eval_x[start_idx:end_idx], dtype=torch.float32).unsqueeze(1).to(device)
            
#             # Predict
#             output = model(batch_x)
            
#             # Convert back to numpy
#             eval_y[start_idx:end_idx] = output.squeeze(1).cpu().numpy()
    
#     end_time_inference = time.time()
#     inference_time = end_time_inference - start_time_inference
    
#     # Postprocessing time
#     start_time_postprocess = time.time()
    
#     # Restore the scale before normalization
#     eval_y = eval_y * (max_x - min_x) + min_x
    
#     # Reconstruct waveform
#     sep_wav, Fs = reconstruct_wave(eval_y, ang_x, Fs, frame_length, frame_shift)
    
#     end_time_postprocess = time.time()
#     postprocess_time = end_time_postprocess - start_time_postprocess
    
#     # Calculate total time
#     end_time_total = time.time()
#     total_processing_time = end_time_total - start_time_total
    
#     # Calculate RTF
#     rtf = inference_time / total_audio_duration
#     rtf_with_processing = total_processing_time / total_audio_duration
    
#     # Save enhanced audio
#     output_file = os.path.join(output_dir, f"enhanced_{os.path.basename(noisy_file)}")
#     sf.write(output_file, sep_wav, Fs)
    
#     # Print detailed timing info
#     print("\n===== INFERENCE STATS =====")
#     print(f"Audio duration: {total_audio_duration:.2f} seconds")
#     print(f"Preprocessing time: {preprocess_time:.2f} seconds")
#     print(f"Neural network inference time: {inference_time:.2f} seconds")
#     print(f"Postprocessing time: {postprocess_time:.2f} seconds")
#     print(f"Total processing time: {total_processing_time:.2f} seconds")
#     print(f"Real-Time Factor (inference only): {rtf:.4f}")
#     print(f"Real-Time Factor (with pre/post processing): {rtf_with_processing:.4f}")
#     print(f"Enhanced audio saved to: {output_file}")
#     print("============================")
    
#     # Try to compute metrics if clean reference is available
#     clean_file = noisy_file.replace("NOISY", "CLEAN")
#     if os.path.exists(clean_file):
#         print("\nComputing audio quality metrics...")
#         try:
#             # Load and preprocess clean reference
#             clean_wav, Fs_clean = sf.read(clean_file)
#             clean_wav, Fs_clean = pre_processing(clean_wav, Fs_clean, down_sample)
            
#             # Adjust lengths
#             min_len = min(len(clean_wav), len(mix_wav), len(sep_wav))
#             clean_wav = clean_wav[:min_len]
#             mix_wav = mix_wav[:min_len]
#             sep_wav = sep_wav[:min_len]
            
#             # Compute metrics
#             pesq_mix = PESQ(Fs, clean_wav, mix_wav, 'wb')
#             stoi_mix = STOI(clean_wav, mix_wav, Fs, extended=False)
#             estoi_mix = STOI(clean_wav, mix_wav, Fs, extended=True)
            
#             pesq_sep = PESQ(Fs, clean_wav, sep_wav, 'wb')
#             stoi_sep = STOI(clean_wav, sep_wav, Fs, extended=False)
#             estoi_sep = STOI(clean_wav, sep_wav, Fs, extended=True)
            
#             print("\n===== QUALITY METRICS =====")
#             print(f"PESQ (noisy): {pesq_mix:.4f}, STOI: {stoi_mix:.4f}, ESTOI: {estoi_mix:.4f}")
#             print(f"PESQ (enhanced): {pesq_sep:.4f}, STOI: {stoi_sep:.4f}, ESTOI: {estoi_sep:.4f}")
#             print(f"Improvement - PESQ: {pesq_sep-pesq_mix:.4f}, STOI: {stoi_sep-stoi_mix:.4f}, ESTOI: {estoi_sep-estoi_mix:.4f}")
#             print("============================")
            
#         except Exception as e:
#             print(f"Error computing metrics: {e}")
    
#     # Save results to log file
#     os.makedirs('./log', exist_ok=True)
#     with open(f"./log/inference_log_{os.path.basename(noisy_file)}.txt", "w") as f:
#         f.write("===== INFERENCE STATS =====\n")
#         f.write(f"Model parameters: {total_params:,}\n")
#         f.write(f"Audio duration: {total_audio_duration:.2f} seconds\n")
#         f.write(f"Preprocessing time: {preprocess_time:.2f} seconds\n")
#         f.write(f"Neural network inference time: {inference_time:.2f} seconds\n")
#         f.write(f"Postprocessing time: {postprocess_time:.2f} seconds\n")
#         f.write(f"Total processing time: {total_processing_time:.2f} seconds\n")
#         f.write(f"Real-Time Factor (inference only): {rtf:.4f}\n")
#         f.write(f"Real-Time Factor (with pre/post processing): {rtf_with_processing:.4f}\n")
#         f.write(f"Enhanced audio saved to: {output_file}\n")
        
#         if os.path.exists(clean_file):
#             try:
#                 f.write("\n===== QUALITY METRICS =====\n")
#                 f.write(f"PESQ (noisy): {pesq_mix:.4f}, STOI: {stoi_mix:.4f}, ESTOI: {estoi_mix:.4f}\n")
#                 f.write(f"PESQ (enhanced): {pesq_sep:.4f}, STOI: {stoi_sep:.4f}, ESTOI: {estoi_sep:.4f}\n")
#                 f.write(f"Improvement - PESQ: {pesq_sep-pesq_mix:.4f}, STOI: {stoi_sep-stoi_mix:.4f}, ESTOI: {estoi_sep-estoi_mix:.4f}\n")
#             except:
#                 f.write("\nError computing metrics\n")
    
#     return rtf, total_audio_duration, inference_time

# ### Main ###
# if __name__ == "__main__":
#     # Check command line arguments
#     if len(sys.argv) > 1:
#         # Use file from command line argument
#         noisy_file = sys.argv[1]
#         if not os.path.exists(noisy_file):
#             print(f"Error: File {noisy_file} not found")
#             sys.exit(1)
        
#         real_time_inference(noisy_file)
#     else:
#         # Use sample files from evaluation directory
#         eval_dir = "./audio_data/evaluation/NOISY"
#         if not os.path.exists(eval_dir):
#             print(f"Error: Evaluation directory {eval_dir} not found")
#             sys.exit(1)
            
#         files = sorted(glob.glob(f"{eval_dir}/*.wav"))
#         if not files:
#             print(f"Error: No WAV files found in {eval_dir}")
#             sys.exit(1)
            
#         # Use first 5 files or all if less than 5
#         test_files = files[:min(5, len(files))]
        
#         # Process each file and track RTF
#         rtf_values = []
        
#         for file in test_files:
#             print(f"\nProcessing file: {os.path.basename(file)}")
#             rtf, duration, infer_time = real_time_inference(file)
#             rtf_values.append(rtf)
            
#         # Calculate average RTF
#         avg_rtf = sum(rtf_values) / len(rtf_values)
#         print(f"\nAverage Real-Time Factor across {len(test_files)} files: {avg_rtf:.4f}")
        
#         if avg_rtf < 1.0:
#             print(f"The model runs {1/avg_rtf:.2f}x faster than real-time")
#         else:
#             print(f"The model runs {avg_rtf:.2f}x slower than real-time")

#!/usr/bin/env python
# coding: utf-8

### Inference script for U-Net speech enhancement ###
import os
import sys
import time
import glob
import math
import random
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

### U-Net model definition in PyTorch ###
class UNetModel(nn.Module):
    def __init__(self):
        super(UNetModel, self).__init__()
        
        # Simplified encoder (just 3 layers)
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
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Simplified decoder (just 3 layers)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3), output_padding=(0, 1)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Decoder with skip connections
        d1 = self.dec1(e3)
        d1 = F.interpolate(d1, size=e2.size()[2:], mode='nearest')
        d2 = self.dec2(torch.cat([d1, e2], dim=1))
        d2 = F.interpolate(d2, size=e1.size()[2:], mode='nearest')
        d3 = self.dec3(torch.cat([d2, e1], dim=1))
        d3 = F.interpolate(d3, size=x.size()[2:], mode='nearest')
        
        return d3

### Function for pre-processing ###
def pre_processing(data, Fs, down_sample):
    
    #Transform stereo into monoral
    if data.ndim == 2:
        wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
    else:
        wavdata = data
    
    #Downsample if necessary
    if down_sample is not None:
        wavdata = sg.resample_poly(wavdata, down_sample, Fs)
        Fs = down_sample
    
    return wavdata, Fs

### Function for reading an audio and getting the STFT ###
def read_evaldata(file_path, down_sample, frame_length, frame_shift, num_frames):
    
    #Initialize list
    x = []
    ang_x = []
    
    #Read .wav file and get pre-process
    wavdata, Fs = sf.read(file_path)
    wavdata, Fs = pre_processing(wavdata, Fs, down_sample)
    
    #Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    
    #Execute STFT
    _, _, dft = sg.stft(wavdata, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
    dft = dft[:-1].T #Remove the last point and get transpose
    ang = np.angle(dft) #Preserve the phase
    spec = np.log10(np.abs(dft) + 1e-10)  # Add small epsilon to avoid log(0)
    
    #Crop the temporal frames into input size
    num_seg = math.floor(spec.shape[0] / num_frames)
    for j in range(num_seg):
        #Add results to list sequentially
        x.append(spec[int(j*num_frames) : int((j+1)*num_frames), :])
        ang_x.append(ang[int(j*num_frames) : int((j+1)*num_frames), :])
    
    #Convert into numpy array
    x = np.array(x)
    ang_x = np.array(ang_x)
    
    return wavdata, Fs, x, ang_x

### Function for reconstructing a waveform ###
def reconstruct_wave(eval_y, ang_x, Fs, frame_length, frame_shift):
    
    #Construct the spectrogram by concatenating all segments
    Y = np.reshape(eval_y, (-1, eval_y.shape[-1]))
    ang = np.reshape(ang_x, (-1, ang_x.shape[-1]))
    
    #The Y and arg can be transpose for processing
    Y, ang = Y.T, ang.T
    
    #Restore the magnitude of STFT
    Y = np.power(10, Y)
    
    #Retrieve the phase from original wave
    Y = Y * np.exp(1j*ang)
    
    #Add the last frequency bin along with frequency axis
    Y = np.append(Y, Y[-1, :][np.newaxis,:], axis=0)
    
    #Get the inverse STFT
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    _, rec_wav = sg.istft(Y, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
    
    return rec_wav, Fs

### Function for measuring time per frame in milliseconds ###
def time_per_frame(model, input_frames, batch_size=1, num_runs=100):
    """Measure average processing time per frame in milliseconds"""
    model.eval()
    
    # Ensure input is on the correct device
    input_tensor = torch.tensor(input_frames, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Warmup runs
    for _ in range(10):
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

### Function for real-time inference with RTF calculation ###
def real_time_inference(noisy_file, output_dir="./audio_data/output", model_path="./models/unet_model.pth"):
    # Set parameters
    down_sample = 16000    # Downsampling rate (Hz)
    frame_length = 0.032   # STFT window width (second)
    frame_shift = 0.016    # STFT window shift (second)
    num_frames = 16        # The number of frames for an input
    
    # Normalization parameters (loaded or default)
    try:
        norm_values = np.load('./numpy_files/norm_values.npy', allow_pickle=True).item()
        min_x = norm_values['min']
        max_x = norm_values['max']
        print(f"Loaded normalization values: min={min_x}, max={max_x}")
    except:
        # Use default values
        min_x = -9.188868273733446
        max_x = -0.2012536819610933
        print(f"Using default normalization values: min={min_x}, max={max_x}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = UNetModel().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total model parameters: {total_params:,}')
    
    # Try to load model weights
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from '{model_path}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try to find checkpoints
        checkpoints = sorted(glob.glob('./models/unet_checkpoint_*.pth'))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Loading latest checkpoint: {latest_checkpoint}")
            model.load_state_dict(torch.load(latest_checkpoint))
        else:
            print("No model or checkpoint found. Using untrained model for proof of concept...")
    
    model.eval()
    
    # Read audio and compute STFT
    start_time_total = time.time()
    
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
    batch_size = 32
    num_batches = math.ceil(eval_x.shape[0] / batch_size)
    eval_y = np.zeros_like(eval_x)
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, eval_x.shape[0])
            
            # Convert batch to tensor
            batch_x = torch.tensor(eval_x[start_idx:end_idx], dtype=torch.float32).unsqueeze(1).to(device)
            
            # Predict
            output = model(batch_x)
            
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
    
    # Calculate total time
    end_time_total = time.time()
    total_processing_time = (end_time_total - start_time_total) * 1000  # milliseconds
    
    # Calculate RTF
    rtf = inference_time / total_audio_duration_ms
    rtf_with_processing = total_processing_time / total_audio_duration_ms
    
    # Save enhanced audio
    output_file = os.path.join(output_dir, f"enhanced_{os.path.basename(noisy_file)}")
    sf.write(output_file, sep_wav, Fs)
    
    # Measure per-frame processing time
    if len(eval_x) > 0:
        print("\nMeasuring per-frame processing time...")
        frame_avg_ms, frame_min_ms, frame_max_ms = time_per_frame(model, eval_x[0:1], batch_size=1, num_runs=100)
    else:
        frame_avg_ms, frame_min_ms, frame_max_ms = 0, 0, 0
    
    # Calculate frames per second
    frames_per_second = 1000 / frame_avg_ms if frame_avg_ms > 0 else 0
    
    # Print detailed timing info
    print("\n===== INFERENCE STATS =====")
    print(f"Audio duration: {total_audio_duration:.2f} seconds ({total_audio_duration_ms:.1f} ms)")
    print(f"Preprocessing time: {preprocess_time:.1f} ms")
    print(f"Neural network inference time: {inference_time:.1f} ms")
    print(f"Postprocessing time: {postprocess_time:.1f} ms")
    print(f"Total processing time: {total_processing_time:.1f} ms")
    print(f"RTF (inference only): {rtf:.4f}")
    print(f"RTF (with pre/post processing): {rtf_with_processing:.4f}")
    print(f"Average processing time per frame: {frame_avg_ms:.2f} ms (min: {frame_min_ms:.2f} ms, max: {frame_max_ms:.2f} ms)")
    print(f"Frames processed per second: {frames_per_second:.1f}")
    print(f"Enhanced audio saved to: {output_file}")
    print("============================")
    
    # Try to compute metrics if clean reference is available
    clean_file = noisy_file.replace("NOISY", "CLEAN")
    if os.path.exists(clean_file):
        print("\nComputing audio quality metrics...")
        try:
            # Load and preprocess clean reference
            clean_wav, Fs_clean = sf.read(clean_file)
            clean_wav, Fs_clean = pre_processing(clean_wav, Fs_clean, down_sample)
            
            # Adjust lengths
            min_len = min(len(clean_wav), len(mix_wav), len(sep_wav))
            clean_wav = clean_wav[:min_len]
            mix_wav = mix_wav[:min_len]
            sep_wav = sep_wav[:min_len]
            
            # Compute metrics
            pesq_mix = PESQ(Fs, clean_wav, mix_wav, 'wb')
            stoi_mix = STOI(clean_wav, mix_wav, Fs, extended=False)
            estoi_mix = STOI(clean_wav, mix_wav, Fs, extended=True)
            
            pesq_sep = PESQ(Fs, clean_wav, sep_wav, 'wb')
            stoi_sep = STOI(clean_wav, sep_wav, Fs, extended=False)
            estoi_sep = STOI(clean_wav, sep_wav, Fs, extended=True)
            
            print("\n===== QUALITY METRICS =====")
            print(f"PESQ (noisy): {pesq_mix:.4f}, STOI: {stoi_mix:.4f}, ESTOI: {estoi_mix:.4f}")
            print(f"PESQ (enhanced): {pesq_sep:.4f}, STOI: {stoi_sep:.4f}, ESTOI: {estoi_sep:.4f}")
            print(f"Improvement - PESQ: {pesq_sep-pesq_mix:.4f}, STOI: {stoi_sep-stoi_mix:.4f}, ESTOI: {estoi_sep-estoi_mix:.4f}")
            print("============================")
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
    
    # Save results to log file
    os.makedirs('./log', exist_ok=True)
    with open(f"./log/inference_log_{os.path.basename(noisy_file)}.txt", "w") as f:
        f.write("===== INFERENCE STATS =====\n")
        f.write(f"Model parameters: {total_params:,}\n")
        f.write(f"Audio duration: {total_audio_duration:.2f} seconds ({total_audio_duration_ms:.1f} ms)\n")
        f.write(f"Preprocessing time: {preprocess_time:.1f} ms\n")
        f.write(f"Neural network inference time: {inference_time:.1f} ms\n")
        f.write(f"Postprocessing time: {postprocess_time:.1f} ms\n")
        f.write(f"Total processing time: {total_processing_time:.1f} ms\n")
        f.write(f"RTF (inference only): {rtf:.4f}\n")
        f.write(f"RTF (with pre/post processing): {rtf_with_processing:.4f}\n")
        f.write(f"Average processing time per frame: {frame_avg_ms:.2f} ms (min: {frame_min_ms:.2f} ms, max: {frame_max_ms:.2f} ms)\n")
        f.write(f"Frames processed per second: {frames_per_second:.1f}\n")
        
        if os.path.exists(clean_file):
            try:
                f.write("\n===== QUALITY METRICS =====\n")
                f.write(f"PESQ (noisy): {pesq_mix:.4f}, STOI: {stoi_mix:.4f}, ESTOI: {estoi_mix:.4f}\n")
                f.write(f"PESQ (enhanced): {pesq_sep:.4f}, STOI: {stoi_sep:.4f}, ESTOI: {estoi_sep:.4f}\n")
                f.write(f"Improvement - PESQ: {pesq_sep-pesq_mix:.4f}, STOI: {stoi_sep-stoi_mix:.4f}, ESTOI: {estoi_sep-estoi_mix:.4f}\n")
            except:
                f.write("\nError computing metrics\n")
    
    return rtf, total_audio_duration, inference_time, frame_avg_ms

### Main ###
if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        # Use file from command line argument
        noisy_file = sys.argv[1]
        if not os.path.exists(noisy_file):
            print(f"Error: File {noisy_file} not found")
            sys.exit(1)
        
        real_time_inference(noisy_file)
    else:
        # Use sample files from evaluation directory
        eval_dir = "./audio_data/evaluation/NOISY"
        if not os.path.exists(eval_dir):
            print(f"Error: Evaluation directory {eval_dir} not found")
            sys.exit(1)
            
        files = sorted(glob.glob(f"{eval_dir}/*.wav"))
        if not files:
            print(f"Error: No WAV files found in {eval_dir}")
            sys.exit(1)
            
        # Use first 5 files or all if less than 5
        test_files = files[:min(5, len(files))]
        
        # Process each file and track metrics
        rtf_values = []
        frame_times = []
        
        for file in test_files:
            print(f"\nProcessing file: {os.path.basename(file)}")
            rtf, duration, infer_time, frame_time = real_time_inference(file)
            rtf_values.append(rtf)
            frame_times.append(frame_time)
            
        # Calculate averages
        avg_rtf = sum(rtf_values) / len(rtf_values)
        avg_frame_time = sum(frame_times) / len(frame_times)
        
        print("\n===== SUMMARY =====")
        print(f"Average Real-Time Factor across {len(test_files)} files: {avg_rtf:.4f}")
        print(f"Average processing time per frame: {avg_frame_time:.2f} ms")
        
        if avg_rtf < 1.0:
            print(f"The model runs {1/avg_rtf:.2f}x faster than real-time")
        else:
            print(f"The model runs {avg_rtf:.2f}x slower than real-time")