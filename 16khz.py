#!/usr/bin/env python3
"""
Convert all audio files from 48kHz to 16kHz
"""

import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def resample_audio_file(input_path, output_path, target_sr=16000):
    """Resample an audio file to target sample rate"""
    try:
        # Load audio
        audio, sr = torchaudio.load(input_path)
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)
        
        # Save resampled audio
        torchaudio.save(output_path, audio, target_sr)
        return True
        
    except Exception as e:
        print(f"Error resampling {input_path}: {e}")
        return False

def resample_dataset(csv_path, output_suffix="_16khz"):
    """Resample all audio files in a CSV dataset"""
    
    df = pd.read_csv(csv_path)
    
    # Create output directories
    Path("data_16khz/training/noisy").mkdir(parents=True, exist_ok=True)
    Path("data_16khz/training/clean").mkdir(parents=True, exist_ok=True)
    Path("data_16khz/evaluation/noisy").mkdir(parents=True, exist_ok=True)
    Path("data_16khz/evaluation/clean").mkdir(parents=True, exist_ok=True)
    
    new_pairs = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Resampling {csv_path}"):
        noisy_path = Path(row['noisy_path'])
        clean_path = Path(row['clean_path'])
        
        # Create new paths with 16kHz suffix
        new_noisy_path = Path("data_16khz") / noisy_path.relative_to("data")
        new_clean_path = Path("data_16khz") / clean_path.relative_to("data")
        
        # Resample both files
        noisy_success = resample_audio_file(noisy_path, new_noisy_path)
        clean_success = resample_audio_file(clean_path, new_clean_path)
        
        if noisy_success and clean_success:
            new_pairs.append({
                'noisy_path': str(new_noisy_path),
                'clean_path': str(new_clean_path)
            })
    
    # Save new CSV
    output_csv = csv_path.replace('.csv', f'{output_suffix}.csv')
    new_df = pd.DataFrame(new_pairs)
    new_df.to_csv(output_csv, index=False)
    
    print(f" Resampled {len(new_pairs)} pairs")
    print(f"New CSV: {output_csv}")
    
    return output_csv

def main():
    print(" Converting audio files to 16kHz...")
    print("=" * 50)
    
    # Resample training set
    train_csv = resample_dataset("data/training_pairs.csv")
    
    # Resample evaluation set  
    eval_csv = resample_dataset("data/evaluation_pairs.csv")
    
    print(f"\n All done! Update your config to use:")
    print(f"trainset:")
    print(f"  csv_path: '{train_csv}'")
    print(f"valset:")
    print(f"  csv_path: '{eval_csv}'")

if __name__ == "__main__":
    main()
