#!/usr/bin/env python3
"""
Script to rename audio files from p234_134.wav format to sample1.wav, sample2.wav, etc.
Run this first to rename your files.
"""

import os
import shutil
from pathlib import Path

def rename_files_in_folder(folder_path, prefix="sample"):
    """
    Rename all .wav files in a folder to sample1.wav, sample2.wav, etc.
    
    Args:
        folder_path: Path to folder containing .wav files
        prefix: Prefix for new names (default: "sample")
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder} does not exist!")
        return
    
    # Get all .wav files and sort them for consistent naming
    wav_files = sorted(list(folder.glob("*.wav")))
    
    if not wav_files:
        print(f"No .wav files found in {folder}")
        return
    
    print(f"Found {len(wav_files)} .wav files in {folder}")
    print("Renaming files...")
    
    # Rename files
    for i, old_file in enumerate(wav_files, 1):
        new_name = f"{prefix}{i}.wav"
        new_path = folder / new_name
        
        # Avoid overwriting if file already exists
        if new_path.exists():
            print(f"Warning: {new_name} already exists, skipping {old_file.name}")
            continue
            
        old_file.rename(new_path)
        if i % 1000 == 0:  # Progress update every 1000 files
            print(f"Renamed {i}/{len(wav_files)} files...")
    
    print(f"✅ Renamed {len(wav_files)} files in {folder}")

def main():
    # Define your current data folders
    noisy_folder = "noisy"  # Change this to your actual noisy folder path
    clean_folder = "clean"  # Change this to your actual clean folder path
    
    print("🔄 Renaming audio files...")
    print("=" * 50)
    
    # Check if folders exist
    if not Path(noisy_folder).exists():
        print(f"❌ Noisy folder '{noisy_folder}' not found!")
        print("Please update the folder paths in this script.")
        return
    
    if not Path(clean_folder).exists():
        print(f"❌ Clean folder '{clean_folder}' not found!")
        print("Please update the folder paths in this script.")
        return
    
    # Rename files in both folders
    print("Renaming noisy files...")
    rename_files_in_folder(noisy_folder, prefix="sample")
    
    print("\nRenaming clean files...")
    rename_files_in_folder(clean_folder, prefix="sample")
    
    print("\n✅ All files renamed successfully!")
    print("You can now run the next script: 2_split_and_create_csvs.py")

if __name__ == "__main__":
    main()
