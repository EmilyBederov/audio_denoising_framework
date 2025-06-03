#!/usr/bin/env python3
"""
Final setup for your pre-organized data with existing CSV files
Just extract and fix the CSV paths
"""

import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path

def extract_data(zip_path="data/audio_data.zip", target_dir="data/audio_data"):
    """
    Extract the zip file to target directory
    """
    project_root = Path(__file__).parent
    zip_full_path = project_root / zip_path
    target_path = project_root / target_dir
    
    if not zip_full_path.exists():
        print(f"ERROR: {zip_path} not found!")
        return False
    
    # Remove existing data
    if target_path.exists():
        print("Removing existing data directory...")
        shutil.rmtree(target_path)
    
    # Extract zip
    print(f"Extracting {zip_path}...")
    target_path.mkdir()
    with zipfile.ZipFile(zip_full_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)
    
    print(f"Data extracted to: {target_path}")
    return True

def clean_macos_files(data_dir="data/audio_data"):
    """
    Remove __MACOSX and .DS_Store files
    """
    data_path = Path(data_dir)
    
    # Remove __MACOSX directory
    macos_dir = data_path / "__MACOSX"
    if macos_dir.exists():
        print("Removing __MACOSX files...")
        shutil.rmtree(macos_dir)
    
    # Remove .DS_Store files
    for ds_store in data_path.rglob(".DS_Store"):
        ds_store.unlink()

def fix_csv_paths(data_dir="data/audio_data"):
    """
    Fix the CSV file paths to point to the correct locations
    """
    data_path = Path(data_dir)
    
    # Process training CSV
    training_csv = data_path / "training_pairs_16khz.csv"
    if training_csv.exists():
        print(f"Processing {training_csv}...")
        df = pd.read_csv(training_csv)
        
        # Update paths to use data/ prefix
        if 'noisy_path' in df.columns and 'clean_path' in df.columns:
            df['noisy_path'] = df['noisy_path'].apply(lambda x: f"data/audio_data/{x}")
            df['clean_path'] = df['clean_path'].apply(lambda x: f"data/audio_data/{x}")
        else:
            # If columns are different, try to identify them
            cols = df.columns.tolist()
            print(f"CSV columns found: {cols}")
            # Assume first column is noisy, second is clean
            df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: f"data/audio_data/{x}")
            df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: f"data/audio_data/{x}")
            # Rename columns to standard names
            df.columns = ['noisy_path', 'clean_path']
        
        # Save updated CSV in same location
        df.to_csv(training_csv, index=False)
        print(f"Updated training CSV: {training_csv}")
        print(f"Training pairs: {len(df)}")
    else:
        print(f"WARNING: {training_csv} not found!")
        return False
    
    # Process evaluation CSV
    eval_csv = data_path / "evaluation_pairs_16khz.csv"
    if eval_csv.exists():
        print(f"Processing {eval_csv}...")
        df = pd.read_csv(eval_csv)
        
        # Update paths to use data/ prefix
        if 'noisy_path' in df.columns and 'clean_path' in df.columns:
            df['noisy_path'] = df['noisy_path'].apply(lambda x: f"data/audio_data/{x}")
            df['clean_path'] = df['clean_path'].apply(lambda x: f"data/audio_data/{x}")
        else:
            # If columns are different, try to identify them
            cols = df.columns.tolist()
            print(f"CSV columns found: {cols}")
            # Assume first column is noisy, second is clean
            df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: f"data/audio_data/{x}")
            df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: f"data/audio_data/{x}")
            # Rename columns to standard names
            df.columns = ['noisy_path', 'clean_path']
        
        # Save updated CSV in same location
        df.to_csv(eval_csv, index=False)
        print(f"Updated evaluation CSV: {eval_csv}")
        print(f"Evaluation pairs: {len(df)}")
    else:
        print(f"WARNING: {eval_csv} not found!")
        return False
    
    return True

def verify_setup():
    """
    Verify everything is set up correctly
    """
    print("\nVerifying setup...")
    
    # Check directories
    required_dirs = [
        "data/audio_data/training",
        "data/audio_data/evaluation"
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            # Count audio files
            wav_files = list(dir_path.rglob("*.wav"))
            print(f"  {dir_name}: {len(wav_files)} wav files")
        else:
            print(f"  {dir_name}: NOT FOUND")
            return False
    
    # Check CSV files
    required_csvs = [
        "data/audio_data/training_pairs_16khz.csv",
        "data/audio_data/evaluation_pairs_16khz.csv"
    ]
    
    for csv_name in required_csvs:
        csv_path = Path(csv_name)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  {csv_name}: {len(df)} pairs")
        else:
            print(f"  {csv_name}: NOT FOUND")
            return False
    
    return True

def main():
    """
    Main setup function
    """
    print("Final Audio Data Setup")
    print("=" * 40)
    
    # Step 1: Extract data
    if not extract_data():
        return False
    
    # Step 2: Clean macOS files
    clean_macos_files()
    
    # Step 3: Fix CSV paths
    if not fix_csv_paths():
        return False
    
    # Step 4: Verify setup
    if not verify_setup():
        print("ERROR: Setup verification failed!")
        return False
    
    # Success summary
    print("\n" + "=" * 40)
    print("Setup Complete!")
    print("Your data and CSV files are ready!")
    print("\nRun: python run.py --config configs/configs-cleanunet2/cleanunet2-config.yaml --model cleanunet2 --mode train")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nSetup failed!")
        exit(1)