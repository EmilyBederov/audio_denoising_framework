import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import random
from torch.utils.data import Dataset, DataLoader

# Import your generator and discriminator models
from Generator import MGAN_G
from Discriminator import Discriminator

# HASQI-related functions - You'll need to implement these based on the HASQI algorithm
# or use an existing implementation
def compute_hasqi(clean_comp, est_speech, audiogram):
    """
    Compute HASQI score between clean compensated speech and estimated speech
    """
    # This is a placeholder - you need to implement the actual HASQI algorithm
    # or use an external library that computes HASQI
    return 0.8  # Placeholder value

# PMSQE and PASE loss functions
class PMSQELoss(nn.Module):
    def __init__(self):
        super(PMSQELoss, self).__init__()
        # Initialize PMSQE parameters
        # This is just a placeholder - you need to implement the actual PMSQE loss

    def forward(self, clean, est):
        # Calculate PMSQE loss between clean and estimated speech
        # This is a placeholder - implement the actual PMSQE calculation
        return torch.mean((clean - est) ** 2)

class PASELoss(nn.Module):
    def __init__(self):
        super(PASELoss, self).__init__()
        # Initialize PASE parameters
        # This is just a placeholder - you need to implement the actual PASE loss

    def forward(self, clean, est):
        # Calculate PASE loss between clean and estimated speech
        # This is a placeholder - implement the actual PASE calculation
        return torch.mean(torch.abs(clean - est))

# Dataset handling
class HearingAidDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, audiogram_file, segment_length=48000, sr=16000):
        """
        Dataset for training hearing aid model
        
        Args:
            clean_dir: Directory with clean speech files
            noisy_dir: Directory with noisy speech files
            audiogram_file: Path to file containing audiograms
            segment_length: Length of audio segments
            sr: Sample rate
        """
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.segment_length = segment_length
        self.sr = sr
        
        # Load clean and noisy file lists
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        
        # Load audiograms
        self.audiograms = {}
        with open(audiogram_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                file_id = parts[0]
                # Convert audiogram thresholds to floats
                thresholds = [float(x) for x in parts[1:]]
                self.audiograms[file_id] = thresholds
                
        # Get all available audiogram keys
        self.audiogram_keys = list(self.audiograms.keys())
        
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # Load clean and noisy audio
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        
        clean_audio, _ = torchaudio.load(clean_path)
        noisy_audio, _ = torchaudio.load(noisy_path)
        
        # Ensure audio is mono
        if clean_audio.shape[0] > 1:
            clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
        if noisy_audio.shape[0] > 1:
            noisy_audio = torch.mean(noisy_audio, dim=0, keepdim=True)
            
        # Cut or pad to segment_length
        if clean_audio.shape[1] >= self.segment_length:
            start = random.randint(0, clean_audio.shape[1] - self.segment_length)
            clean_audio = clean_audio[:, start:start+self.segment_length]
            noisy_audio = noisy_audio[:, start:start+self.segment_length]
        else:
            # Pad with zeros
            clean_pad = torch.zeros(1, self.segment_length)
            noisy_pad = torch.zeros(1, self.segment_length)
            clean_pad[:, :clean_audio.shape[1]] = clean_audio
            noisy_pad[:, :noisy_audio.shape[1]] = noisy_audio
            clean_audio = clean_pad
            noisy_audio = noisy_pad
            
        # Randomly select an audiogram
        audiogram_key = random.choice(self.audiogram_keys)
        audiogram = torch.tensor(self.audiograms[audiogram_key])
        
        return clean_audio, noisy_audio, audiogram

# Function to extend audiogram as described in the paper
def extend_audiogram(audiogram, n_freq_bins=257):
    """
    Extend audiogram along frequency axis to match spectrogram dimensions
    Based on Table I in the paper
    
    Args:
        audiogram: Tensor of shape [6] with hearing loss thresholds at 
                   [250, 500, 1000, 2000, 4000, 8000] Hz
        n_freq_bins: Number of frequency bins in the STFT
        
    Returns:
        Extended audiogram of shape [n_freq_bins]
    """
    # Create empty extended audiogram
    extended = torch.zeros(n_freq_bins)
    
    # Map audiogram values to frequency bins based on Table I in the paper
    # These ranges are for 512-point FFT with 16kHz sample rate
    # Adjust if using different parameters
    extended[0:8] = audiogram[0]      # 0-250 Hz
    extended[8:16] = audiogram[1]     # 250-500 Hz
    extended[16:32] = audiogram[2]    # 500-1000 Hz
    extended[32:64] = audiogram[3]    # 1000-2000 Hz
    extended[64:128] = audiogram[4]   # 2000-4000 Hz
    extended[128:] = audiogram[5]     # 4000-8000 Hz
    
    # Normalize to range [0, 1] - may need adjustment based on your specific implementation
    extended = extended / 120.0  # Assuming max hearing loss is 120 dB
    
    return extended

# FIG6 compensation function (simplified version)
def fig6_compensation(clean_spec, audiogram_extended):
    """
    Apply FIG6 formula to compensate for hearing loss
    
    Args:
        clean_spec: Clean speech spectrogram
        audiogram_extended: Extended audiogram
        
    Returns:
        Compensated clean speech spectrogram
    """
    # This is a simplified version - implement the actual FIG6 formula
    # The basic idea is to apply gain based on the audiogram
    gain = audiogram_extended.unsqueeze(0).unsqueeze(0)  # [1, 1, F]
    compensated = clean_spec * (1.0 + gain)
    return compensated

# STFT and iSTFT functions
def stft(x, n_fft=512, hop_length=256, win_length=512, window=None):
    """
    Perform STFT and return complex spectrogram
    """
    if window is None:
        window = torch.hann_window(win_length).to(x.device)
    
    # [B, 1, T] -> [B, F, T, 2]
    complex_spec = torch.stft(
        x.squeeze(1), 
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=False
    )
    
    # Split real and imaginary parts
    real = complex_spec[..., 0]
    imag = complex_spec[..., 1]
    
    return real, imag

def istft(real, imag, n_fft=512, hop_length=256, win_length=512, window=None):
    """
    Perform iSTFT from real and imaginary parts
    """
    if window is None:
        window = torch.hann_window(win_length).to(real.device)
    
    # Combine real and imaginary parts
    complex_spec = torch.stack([real, imag], dim=-1)
    
    # Convert to time domain
    waveform = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )
    
    return waveform.unsqueeze(1)  # [B, 1, T]

# Training function
def train_model(
    clean_dir,
    noisy_dir,
    audiogram_file,
    model_save_dir,
    batch_size=32,
    epochs=20,
    lr_g=0.0006,
    lr_d=0.0006,
    n_fft=512,
    hop_length=256,
    win_length=512,
    segment_length=3 * 16000,  # 3 seconds at 16kHz
    lambda_pmsqe=1.0,
    mu_pase=0.25,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the FTCRN-based Metric GAN model
    
    Args:
        clean_dir: Directory with clean speech files
        noisy_dir: Directory with noisy speech files
        audiogram_file: File containing audiograms
        model_save_dir: Directory to save checkpoints
        batch_size: Batch size
        epochs: Number of epochs
        lr_g: Learning rate for generator
        lr_d: Learning rate for discriminator
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        segment_length: Audio segment length
        lambda_pmsqe: Weight for PMSQE loss
        mu_pase: Weight for PASE loss
        device: Training device
    """
    # Create save directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = HearingAidDataset(
        clean_dir=clean_dir,
        noisy_dir=noisy_dir,
        audiogram_file=audiogram_file,
        segment_length=segment_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    generator = MGAN_G().to(device)
    discriminator = Discriminator(ndf=16, in_channel=3).to(device)
    
    # Initialize optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d)
    
    # Initialize loss functions
    pmsqe_loss = PMSQELoss().to(device)
    pase_loss = PASELoss().to(device)
    mse_loss = nn.MSELoss()
    
    # For STFT
    window = torch.hann_window(win_length).to(device)
    
    # Training loop
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        # Metrics for tracking
        running_g_loss = 0.0
        running_d_loss = 0.0
        running_hasqi = 0.0
        
        progress_bar = tqdm(dataloader)
        for i, (clean_audio, noisy_audio, audiogram) in enumerate(progress_bar):
            # Move data to device
            clean_audio = clean_audio.to(device)
            noisy_audio = noisy_audio.to(device)
            audiogram = audiogram.to(device)
            
            batch_size = clean_audio.shape[0]
            
            # Compute STFT
            clean_real, clean_imag = stft(clean_audio, n_fft, hop_length, win_length, window)
            noisy_real, noisy_imag = stft(noisy_audio, n_fft, hop_length, win_length, window)
            
            # Get extended audiogram
            audiogram_extended = torch.zeros(batch_size, 1, 1, n_fft//2 + 1).to(device)
            for b in range(batch_size):
                extended = extend_audiogram(audiogram[b])
                audiogram_extended[b, 0, :, :] = extended.unsqueeze(0)
            
            # Create time-extended audiogram embedding (replicate along time axis)
            T = noisy_real.shape[2]  # Number of time frames
            audiogram_embed = audiogram_extended.repeat(1, 1, T, 1)
            
            # Apply FIG6 compensation to clean speech (this is the target)
            clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
            clean_phase = torch.atan2(clean_imag, clean_real)
            
            # Compensate clean magnitude
            clean_comp_mag = fig6_compensation(clean_mag, audiogram_extended.squeeze(1))
            
            # Convert back to real/imaginary
            clean_comp_real = clean_comp_mag * torch.cos(clean_phase)
            clean_comp_imag = clean_comp_mag * torch.sin(clean_phase)
            
            # Prepare noisy input for generator
            noisy_spec = torch.stack([noisy_real, noisy_imag], dim=1)  # [B, 2, T, F]
            
            #############################
            # Train Discriminator
            #############################
            discriminator.zero_grad()
            
            # Forward pass through generator
            est_real, est_imag = generator(noisy_spec, audiogram_embed)
            
            # Compute magnitude of estimated and clean compensated speech
            est_mag = torch.sqrt(est_real**2 + est_imag**2)
            clean_comp_mag = torch.sqrt(clean_comp_real**2 + clean_comp_imag**2)
            
            # Compute HASQI scores for each sample in batch
            hasqi_scores = torch.zeros(batch_size, 1).to(device)
            for b in range(batch_size):
                # Convert to CPU for HASQI computation (if needed)
                est_audio = istft(est_real[b].unsqueeze(0), est_imag[b].unsqueeze(0), 
                                 n_fft, hop_length, win_length, window)
                clean_comp_audio = istft(clean_comp_real[b].unsqueeze(0), clean_comp_imag[b].unsqueeze(0), 
                                        n_fft, hop_length, win_length, window)
                
                hasqi = compute_hasqi(clean_comp_audio.cpu().numpy(), 
                                     est_audio.cpu().numpy(), 
                                     audiogram[b].cpu().numpy())
                hasqi_scores[b, 0] = hasqi
            
            # Get discriminator predictions
            est_input = torch.cat([est_mag.unsqueeze(1), clean_comp_mag.unsqueeze(1), audiogram_embed], dim=1)
            est_d_score = discriminator(est_mag, clean_comp_mag, audiogram_embed)
            
            # Set ideal case (both inputs are clean compensated speech)
            ideal_d_score = discriminator(clean_comp_mag, clean_comp_mag, audiogram_embed)
            
            # Discriminator loss as in Equation (2)
            d_loss = mse_loss(ideal_d_score, torch.ones_like(ideal_d_score)) + \
                     mse_loss(est_d_score, hasqi_scores)
            
            d_loss.backward()
            optimizer_d.step()
            
            #############################
            # Train Generator
            #############################
            generator.zero_grad()
            
            # Forward pass through generator
            est_real, est_imag = generator(noisy_spec, audiogram_embed)
            
            # Compute magnitude of estimated speech
            est_mag = torch.sqrt(est_real**2 + est_imag**2)
            
            # Freeze discriminator
            for param in discriminator.parameters():
                param.requires_grad = False
                
            # Get discriminator score
            est_d_score = discriminator(est_mag, clean_comp_mag, audiogram_embed)
            
            # Convert to time domain for computing perceptual losses
            est_audio = istft(est_real, est_imag, n_fft, hop_length, win_length, window)
            clean_comp_audio = istft(clean_comp_real, clean_comp_imag, n_fft, hop_length, win_length, window)
            
            # Compute losses
            g_adv_loss = mse_loss(est_d_score, torch.ones_like(est_d_score))
            g_pmsqe = pmsqe_loss(clean_comp_audio, est_audio)
            g_pase = pase_loss(clean_comp_audio, est_audio)
            
            # Total generator loss as in Equation (1)
            g_loss = g_adv_loss + lambda_pmsqe * g_pmsqe + mu_pase * g_pase
            
            g_loss.backward()
            optimizer_g.step()
            
            # Unfreeze discriminator
            for param in discriminator.parameters():
                param.requires_grad = True
            
            # Update metrics
            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            running_hasqi += torch.mean(hasqi_scores).item()
            
            # Update progress bar
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs} "
                f"G_loss: {running_g_loss/(i+1):.4f} "
                f"D_loss: {running_d_loss/(i+1):.4f} "
                f"HASQI: {running_hasqi/(i+1):.4f}"
            )
            
        # Compute epoch metrics
        epoch_g_loss = running_g_loss / len(dataloader)
        epoch_d_loss = running_d_loss / len(dataloader)
        epoch_hasqi = running_hasqi / len(dataloader)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"G_loss: {epoch_g_loss:.4f} - "
              f"D_loss: {epoch_d_loss:.4f} - "
              f"HASQI: {epoch_hasqi:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'g_loss': epoch_g_loss,
            'd_loss': epoch_d_loss,
            'hasqi': epoch_hasqi
        }, os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
    # Save final model
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict()
    }, os.path.join(model_save_dir, 'final_model.pt'))
    
    print("Training completed!")
    return generator, discriminator

# Example usage
if __name__ == "__main__":
    # Set paths and parameters
    clean_dir = "path/to/clean/audio"
    noisy_dir = "path/to/noisy/audio"
    audiogram_file = "HL_Audiograms.txt"
    model_save_dir = "checkpoints"
    
    # Train model
    generator, discriminator = train_model(
        clean_dir=clean_dir,
        noisy_dir=noisy_dir,
        audiogram_file=audiogram_file,
        model_save_dir=model_save_dir,
        batch_size=32,
        epochs=20,
        lambda_pmsqe=1.0,
        mu_pase=0.25
    )
