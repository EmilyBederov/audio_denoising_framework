# core/data/data_loader.py
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
import gdown

class GoogleDriveDownloader:
    """
    Utility class to download data from Google Drive
    """
    def __init__(self, output_dir="./data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def download_file(self, file_id, destination):
        """
        Download a file from Google Drive using gdown
        
        Args:
            file_id: Google Drive file ID
            destination: Destination path
            
        Returns:
            Path to downloaded file
        """
        output_path = os.path.join(self.output_dir, destination)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        return output_path
    
    def download_folder(self, folder_id, destination=""):
        """
        Download a folder from Google Drive using gdown
        
        Args:
            folder_id: Google Drive folder ID
            destination: Destination subfolder within output_dir
            
        Returns:
            Path to downloaded folder
        """
        output_path = os.path.join(self.output_dir, destination)
        os.makedirs(output_path, exist_ok=True)
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=output_path, quiet=False)
        return output_path

class AudioPairDataset(Dataset):
    """
    Dataset for audio denoising that loads paired clean and noisy audio files
    specified in a CSV file.
    """
    
    def __init__(self, csv_path, sample_rate=16000, n_fft=1024, hop_length=256, 
                 win_length=1024, frame_length=0.032, frame_shift=0.016, num_frames=16,
                 segment_length=None, normalize=True):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file with 'noisy' and 'clean' columns
            sample_rate: Target sample rate for audio
            n_fft: FFT size for spectrogram
            hop_length: Hop length for spectrogram
            win_length: Window length for spectrogram
            frame_length: STFT window width in seconds
            frame_shift: STFT window shift in seconds
            num_frames: Number of frames for an input segment
            segment_length: Length of audio segments in samples (if None, uses full audio)
            normalize: Whether to normalize spectrograms
        """
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.num_frames = num_frames
        self.segment_length = segment_length
        self.normalize = normalize
        
        # Calculate window parameters in samples
        self.FL = round(frame_length * sample_rate)
        self.FS = round(frame_shift * sample_rate)
        self.OL = self.FL - self.FS
        
        # If normalization parameters are not provided, estimate them from the first batch
        self.min_val = None
        self.max_val = None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get file paths
        row = self.df.iloc[idx]
        clean_path = row['clean']
        noisy_path = row['noisy']
        
        # Load audio files
        clean_audio, sr_clean = torchaudio.load(clean_path)
        noisy_audio, sr_noisy = torchaudio.load(noisy_path)
        
        # Preprocess audio
        clean_audio = self._preprocess_audio(clean_audio, sr_clean)
        noisy_audio = self._preprocess_audio(noisy_audio, sr_noisy)
        
        # Ensure both have the same length
        min_length = min(clean_audio.shape[1], noisy_audio.shape[1])
        clean_audio = clean_audio[:, :min_length]
        noisy_audio = noisy_audio[:, :min_length]
        
        # Random segment if segment_length is specified
        if self.segment_length is not None and min_length > self.segment_length:
            start = torch.randint(0, min_length - self.segment_length + 1, (1,)).item()
            clean_audio = clean_audio[:, start:start + self.segment_length]
            noisy_audio = noisy_audio[:, start:start + self.segment_length]
        
        # Convert to numpy for STFT calculation
        clean_np = clean_audio.numpy().squeeze()
        noisy_np = noisy_audio.numpy().squeeze()
        
        # Calculate STFT
        _, _, clean_stft = signal.stft(clean_np, fs=self.sample_rate, window='hann', 
                                     nperseg=self.FL, noverlap=self.OL)
        _, _, noisy_stft = signal.stft(noisy_np, fs=self.sample_rate, window='hann', 
                                     nperseg=self.FL, noverlap=self.OL)
        
        # Remove last frequency point and transpose
        clean_stft = clean_stft[:-1].T  # [time, freq]
        noisy_stft = noisy_stft[:-1].T  # [time, freq]
        
        # Get phase information
        noisy_phase = np.angle(noisy_stft)
        
        # Get magnitude spectrogram
        clean_mag = np.log10(np.abs(clean_stft) + 1e-8)
        noisy_mag = np.log10(np.abs(noisy_stft) + 1e-8)
        
        # Crop to num_frames
        num_segments = min(clean_mag.shape[0], noisy_mag.shape[0]) // self.num_frames
        
        if num_segments == 0:
            # File too short, pad with zeros
            clean_pad = np.zeros((self.num_frames, clean_mag.shape[1]))
            noisy_pad = np.zeros((self.num_frames, noisy_mag.shape[1]))
            
            clean_pad[:clean_mag.shape[0], :] = clean_mag[:min(clean_mag.shape[0], self.num_frames), :]
            noisy_pad[:noisy_mag.shape[0], :] = noisy_mag[:min(noisy_mag.shape[0], self.num_frames), :]
            
            clean_mag = clean_pad
            noisy_mag = noisy_pad
            noisy_phase_pad = np.zeros((self.num_frames, noisy_phase.shape[1]))
            noisy_phase_pad[:noisy_phase.shape[0], :] = noisy_phase[:min(noisy_phase.shape[0], self.num_frames), :]
            noisy_phase = noisy_phase_pad
        else:
            # Randomly select a segment
            start_idx = np.random.randint(0, num_segments) * self.num_frames
            end_idx = start_idx + self.num_frames
            
            clean_mag = clean_mag[start_idx:end_idx, :]
            noisy_mag = noisy_mag[start_idx:end_idx, :]
            noisy_phase = noisy_phase[start_idx:end_idx, :]
        
        # Normalize if needed
        if self.normalize:
            if self.min_val is None or self.max_val is None:
                # First estimate from data
                self.min_val = noisy_mag.min()
                self.max_val = noisy_mag.max()
            
            # Apply normalization
            clean_mag = (clean_mag - self.min_val) / (self.max_val - self.min_val)
            noisy_mag = (noisy_mag - self.min_val) / (self.max_val - self.min_val)
        
        # Convert to torch tensors
        clean_mag = torch.tensor(clean_mag, dtype=torch.float32).unsqueeze(0)  # [1, time, freq]
        noisy_mag = torch.tensor(noisy_mag, dtype=torch.float32).unsqueeze(0)  # [1, time, freq]
        noisy_phase = torch.tensor(noisy_phase, dtype=torch.float32)
        
        return clean_audio, clean_mag, noisy_audio, noisy_mag, noisy_phase
    
    def _preprocess_audio(self, audio, sr):
        """
        Preprocess audio by converting to mono and resampling
        
        Args:
            audio: Audio tensor [channels, samples]
            sr: Sample rate
            
        Returns:
            Preprocessed audio tensor [1, samples]
        """
        # Convert to mono if needed
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        
        return audio
    
    def get_normalization_params(self):
        """Get the min/max values used for normalization"""
        return self.min_val, self.max_val

def load_dataset(csv_path, config):
    """
    Load a dataset from a CSV file
    
    Args:
        csv_path: Path to CSV file
        config: Configuration dictionary
        
    Returns:
        AudioPairDataset
    """
    return AudioPairDataset(
        csv_path=csv_path,
        sample_rate=config.get('sample_rate', 16000),
        n_fft=config.get('n_fft', 1024),
        hop_length=config.get('hop_length', 256),
        win_length=config.get('win_length', 1024),
        frame_length=config.get('frame_length', 0.032),
        frame_shift=config.get('frame_shift', 0.016),
        num_frames=config.get('num_frames', 16),
        segment_length=config.get('segment_length', None),
        normalize=config.get('normalize', True)
    )

def get_dataloaders(csv_path, config, batch_size=32, num_workers=4, val_split=0.1, test_split=0.1):
    """
    Create DataLoader objects for training, validation, and test sets
    
    Args:
        csv_path: Path to CSV file with paired data
        config: Configuration dictionary
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        val_split: Validation split ratio
        test_split: Test split ratio
        
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoader objects
    """
    from torch.utils.data import DataLoader, random_split
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if it has a 'split' column
    if 'split' in df.columns:
        # Split by column values
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
        
        # If any split is empty, allocate from train
        if len(val_df) == 0 or len(test_df) == 0:
            # Shuffle train
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Calculate sizes
            val_size = max(int(len(df) * val_split), len(val_df))
            test_size = max(int(len(df) * test_split), len(test_df))
            train_size = len(df) - val_size - test_size
            
            # Re-split if needed
            if len(val_df) == 0:
                val_df = train_df.iloc[:val_size]
                train_df = train_df.iloc[val_size:]
            
            if len(test_df) == 0:
                test_df = train_df.iloc[:test_size]
                train_df = train_df.iloc[test_size:]
    else:
        # Split randomly
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate sizes
        val_size = int(len(df) * val_split)
        test_size = int(len(df) * test_split)
        train_size = len(df) - val_size - test_size
        
        # Split
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size+val_size]
        test_df = df.iloc[train_size+val_size:]
    
    # Save split CSVs for reference
    data_dir = os.path.dirname(csv_path)
    train_df.to_csv(os.path.join(data_dir, 'train_pairs.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'val_pairs.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test_pairs.csv'), index=False)
    
    # Create datasets
    train_dataset = load_dataset(os.path.join(data_dir, 'train_pairs.csv'), config)
    val_dataset = load_dataset(os.path.join(data_dir, 'val_pairs.csv'), config)
    test_dataset = load_dataset(os.path.join(data_dir, 'test_pairs.csv'), config)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def download_and_prepare_data(gdrive_folder_id, output_dir="./data"):
    """
    Download audio data from Google Drive and prepare CSV file
    
    Args:
        gdrive_folder_id: Google Drive folder ID
        output_dir: Output directory
        
    Returns:
        Path to CSV file with paired data
    """
    # Download data from Google Drive
    downloader = GoogleDriveDownloader(output_dir=output_dir)
    downloader.download_folder(gdrive_folder_id)
    
    # Create paired CSV file
    pairs = []
    
    # Check for training data
    train_clean_dir = os.path.join(output_dir, 'train', 'clean')
    train_noisy_dir = os.path.join(output_dir, 'train', 'noisy')
    
    if os.path.exists(train_clean_dir) and os.path.exists(train_noisy_dir):
        for file in os.listdir(train_noisy_dir):
            if file.endswith('.wav'):
                noisy_path = os.path.join(train_noisy_dir, file)
                clean_path = os.path.join(train_clean_dir, file)
                
                if os.path.exists(clean_path):
                    pairs.append({
                        'clean': clean_path,
                        'noisy': noisy_path,
                        'split': 'train'
                    })
    
    # Check for evaluation data
    eval_clean_dir = os.path.join(output_dir, 'eval', 'clean')
    eval_noisy_dir = os.path.join(output_dir, 'eval', 'noisy')
    
    if os.path.exists(eval_clean_dir) and os.path.exists(eval_noisy_dir):
        for file in os.listdir(eval_noisy_dir):
            if file.endswith('.wav'):
                noisy_path = os.path.join(eval_noisy_dir, file)
                clean_path = os.path.join(eval_clean_dir, file)
                
                if os.path.exists(clean_path):
                    # Split evaluation data equally between val and test
                    split = 'val' if len(pairs) % 2 == 0 else 'test'
                    
                    pairs.append({
                        'clean': clean_path,
                        'noisy': noisy_path,
                        'split': split
                    })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(pairs)
    csv_path = os.path.join(output_dir, 'audio_pairs.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Created CSV file with {len(df)} pairs")
    print(f"Split: {df['split'].value_counts().to_dict()}")
    
    return csv_path