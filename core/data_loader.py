# core/data_loader.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from pathlib import Path

class DenoisingDataset(Dataset):
    """Dataset for audio denoising using CSV files - based on working CleanUNet2 implementation"""
    
    def __init__(self, csv_path, sample_rate=16000, n_fft=1024, hop_length=256, win_length=1024, power=1.0, crop_length_sec=4.0):
        """
        Initialize dataset
        
        Args:
            csv_path: Path to CSV file with columns 'noisy_path' and 'clean_path'
            sample_rate: Target sample rate
            n_fft: FFT size for spectrogram
            hop_length: Hop length for spectrogram
            win_length: Window length for spectrogram
            power: Power for spectrogram (1.0 for magnitude)
            crop_length_sec: Crop length in seconds (0 = no cropping)
        """
        self.csv_path = csv_path
        self.sample_rate = sample_rate
        self.crop_length = int(crop_length_sec * sample_rate) if crop_length_sec > 0 else 0
        
        # Create spectrogram transform
        self.spectrogram_fn = T.Spectrogram(
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            power=power, 
            normalized=True, 
            center=False
        )
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Check what columns exist and find the right ones
        print(f"CSV columns: {list(self.df.columns)}")
        
        # Try different possible column names
        noisy_col = None
        clean_col = None
        
        for col in self.df.columns:
            if 'noisy' in col.lower():
                noisy_col = col
            elif 'clean' in col.lower():
                clean_col = col
        
        if noisy_col is None or clean_col is None:
            raise ValueError(f"Could not find noisy and clean columns in CSV. Available columns: {list(self.df.columns)}")
        
        self.noisy_col = noisy_col
        self.clean_col = clean_col
        
        # Create file pairs list
        self.files = list(zip(self.df[noisy_col], self.df[clean_col]))
        
        print(f"Using columns: noisy='{noisy_col}', clean='{clean_col}'")
        print(f"Loaded {len(self.files)} audio pairs")
    
    def __len__(self):
        return len(self.files)
    
    def _load_audio_and_resample(self, filepath: str, target_sr: int) -> torch.Tensor:
        """Load audio file and resample to target sample rate"""
        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
            return torch.zeros(target_sr)  # return 1 second of silence

        # Resample if necessary
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        return waveform
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        
        Returns:
            tuple: (clean_audio, clean_spec, noisy_audio, noisy_spec)
        """
        noisy_path, clean_path = self.files[index]

        # Load audio files
        noisy = self._load_audio_and_resample(noisy_path, self.sample_rate)
        clean = self._load_audio_and_resample(clean_path, self.sample_rate)

        # Crop (optional)
        if self.crop_length > 0:
            crop_len = min(len(noisy), len(clean))
            if self.crop_length < crop_len:
                start = torch.randint(0, crop_len - self.crop_length + 1, (1,)).item()
                noisy = noisy[start:start + self.crop_length]
                clean = clean[start:start + self.crop_length]

        # Align lengths
        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]

        # Pad to valid STFT length
        n_fft = self.spectrogram_fn.n_fft
        hop = self.spectrogram_fn.hop_length
        total_len = ((min_len - n_fft) // hop + 1) * hop + n_fft
        pad = max(0, total_len - min_len)
        noisy = F.pad(noisy, (0, pad))
        clean = F.pad(clean, (0, pad))

        # Compute spectrograms
        noisy_spec = self.spectrogram_fn(noisy.unsqueeze(0))  # → [F, T]
        clean_spec = self.spectrogram_fn(clean.unsqueeze(0))  # → [F, T]

        # Return: (clean_audio, clean_spec, noisy_audio, noisy_spec)
        return clean.unsqueeze(0), clean_spec, noisy.unsqueeze(0), noisy_spec

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    clean_audios, clean_specs, noisy_audios, noisy_specs = zip(*batch)

    # Pad audios to max length in batch
    max_len = max(a.shape[-1] for a in clean_audios)
    
    def pad1d(t, L): 
        return F.pad(t, (0, L - t.shape[-1]))

    clean_audio_batch = torch.stack([pad1d(x, max_len) for x in clean_audios])
    noisy_audio_batch = torch.stack([pad1d(x, max_len) for x in noisy_audios])

    # Pad spectrograms to max T
    max_T = max(s.shape[-1] for s in clean_specs)
    
    def pad2d(t, T): 
        return F.pad(t, (0, T - t.shape[-1]))

    clean_spec_batch = torch.stack([pad2d(x, max_T) for x in clean_specs])  # [B, F, T]
    noisy_spec_batch = torch.stack([pad2d(x, max_T) for x in noisy_specs])  # [B, F, T]

    return clean_audio_batch, clean_spec_batch, noisy_audio_batch, noisy_spec_batch

def get_dataloader(config, split='train'):
    """
    Get dataloader for specified split
    
    Args:
        config: Configuration dictionary
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        DataLoader instance
    """
    if split == 'train':
        csv_path = config['trainset']['csv_path']
    elif split in ['val', 'eval']:
        csv_path = config['valset']['csv_path']
    elif split == 'test':
        # Handle test set - use val if test not specified
        csv_path = config.get('testset', {}).get('csv_path') or config['valset']['csv_path']
    else:
        raise ValueError(f"Unsupported split: {split}")

    # Get audio processing parameters from config
    sample_rate = config.get('sample_rate', 16000)
    n_fft = config.get('n_fft', 1024)
    hop_length = config.get('hop_length', 256)
    win_length = config.get('win_length', 1024)
    power = config.get('power', 1.0)
    crop_length_sec = config.get('max_length', 65536) / sample_rate if config.get('max_length') else 4.0
    
    # Create dataset
    dataset = DenoisingDataset(
        csv_path=csv_path, 
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        crop_length_sec=crop_length_sec
    )

    # Get dataloader parameters
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 4)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=collate_fn  # Use custom collate function
    )
