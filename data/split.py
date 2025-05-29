#!/usr/bin/env python3
"""
Script to:
1. Split renamed files into training (80%) and evaluation (20%) sets
2. Create folder structure: data/training/{noisy,clean}/ and data/evaluation/{noisy,clean}/
3. Generate CSV files with file pairs
"""

import os
import shutil
import pandas as pd
import random
from pathlib import Path

def create_folder_structure():
    """Create the required folder structure"""
    folders = [
        "data/training/noisy",
        "data/training/clean", 
        "data/evaluation/noisy",
        "data/evaluation/clean"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder}")

def split_and_copy_files(source_noisy, source_clean, train_split=0.8):
    """
    Split files into training and evaluation sets and copy them
    
    Args:
        source_noisy: Path to source noisy folder
        source_clean: Path to source clean folder  
        train_split: Fraction for training (default 0.8 = 80%)
    """
    source_noisy = Path(source_noisy)
    source_clean = Path(source_clean)
    
    # Get all sample files (should be sample1.wav, sample2.wav, etc.)
    noisy_files = sorted(list(source_noisy.glob("sample*.wav")))
    clean_files = sorted(list(source_clean.glob("sample*.wav")))
    
    print(f"Found {len(noisy_files)} noisy files and {len(clean_files)} clean files")
    
    # Make sure we have matching files
    if len(noisy_files) != len(clean_files):
        print("  Warning: Number of noisy and clean files don't match!")
    
    # Create list of file pairs
    file_pairs = []
    for i in range(min(len(noisy_files), len(clean_files))):
        noisy_file = noisy_files[i]
        clean_file = clean_files[i]
        
        # Check if they have the same base name
        if noisy_file.stem == clean_file.stem:
            file_pairs.append((noisy_file, clean_file))
        else:
            print(f"️  Skipping mismatched pair: {noisy_file.name} vs {clean_file.name}")
    
    print(f"Found {len(file_pairs)} matching file pairs")
    
    # Shuffle for random split
    random.shuffle(file_pairs)
    
    # Calculate split point
    num_train = int(len(file_pairs) * train_split)
    train_pairs = file_pairs[:num_train]
    eval_pairs = file_pairs[num_train:]
    
    print(f"Split: {len(train_pairs)} training, {len(eval_pairs)} evaluation")
    
    # Copy training files
    print("Copying training files...")
    train_data = []
    for i, (noisy_file, clean_file) in enumerate(train_pairs):
        # Copy noisy file
        dest_noisy = Path("data/training/noisy") / noisy_file.name
        shutil.copy2(noisy_file, dest_noisy)
        
        # Copy clean file  
        dest_clean = Path("data/training/clean") / clean_file.name
        shutil.copy2(clean_file, dest_clean)
        
        # Record for CSV
        train_data.append({
            'noisy_path': f"data/training/noisy/{noisy_file.name}",
            'clean_path': f"data/training/clean/{clean_file.name}"
        })
        
        if (i + 1) % 1000 == 0:
            print(f"Copied {i + 1}/{len(train_pairs)} training files...")
    
    # Copy evaluation files
    print("Copying evaluation files...")
    eval_data = []
    for i, (noisy_file, clean_file) in enumerate(eval_pairs):
        # Copy noisy file
        dest_noisy = Path("data/evaluation/noisy") / noisy_file.name
        shutil.copy2(noisy_file, dest_noisy)
        
        # Copy clean file
        dest_clean = Path("data/evaluation/clean") / clean_file.name  
        shutil.copy2(clean_file, dest_clean)
        
        # Record for CSV
        eval_data.append({
            'noisy_path': f"data/evaluation/noisy/{noisy_file.name}",
            'clean_path': f"data/evaluation/clean/{clean_file.name}"
        })
        
        if (i + 1) % 1000 == 0:
            print(f"Copied {i + 1}/{len(eval_pairs)} evaluation files...")
    
    return train_data, eval_data

def create_csv_files(train_data, eval_data):
    """Create CSV files with file pairs"""
    
    # Create training CSV
    train_df = pd.DataFrame(train_data)
    train_csv_path = "data/training_pairs.csv"
    train_df.to_csv(train_csv_path, index=False)
    print(f" Created {train_csv_path} with {len(train_data)} pairs")
    
    # Create evaluation CSV
    eval_df = pd.DataFrame(eval_data) 
    eval_csv_path = "data/evaluation_pairs.csv"
    eval_df.to_csv(eval_csv_path, index=False)
    print(f" Created {eval_csv_path} with {len(eval_data)} pairs")
    
    # Show sample of CSV content
    print("\nSample from training_pairs.csv:")
    print(train_df.head())
    
    print("\nSample from evaluation_pairs.csv:")
    print(eval_df.head())

def main():
    # Set random seed for reproducible splits
    random.seed(42)
    
    print(" Splitting files and creating CSV pairs...")
    print("=" * 50)
    
    # Define source folders (where your renamed files are)
    source_noisy = "noisy"  # Change this to your actual renamed noisy folder
    source_clean = "clean"  # Change this to your actual renamed clean folder
    
    # Check if source folders exist
    if not Path(source_noisy).exists():
        print(f" Source noisy folder '{source_noisy}' not found!")
        print("Please run 1_rename_files.py first or update the folder paths.")
        return
    
    if not Path(source_clean).exists():
        print(f" Source clean folder '{source_clean}' not found!")
        print("Please run 1_rename_files.py first or update the folder paths.")
        return
    
    # Step 1: Create folder structure
    print("Step 1: Creating folder structure...")
    create_folder_structure()
    
    # Step 2: Split and copy files
    print("\nStep 2: Splitting and copying files...")
    train_data, eval_data = split_and_copy_files(source_noisy, source_clean)
    
    # Step 3: Create CSV files
    print("\nStep 3: Creating CSV files...")
    create_csv_files(train_data, eval_data)
    
    print("\n All done! Your data is ready for training.")
    print("\nFolder structure created:")
    print("data/")
    print("├── training/")
    print("│   ├── noisy/")
    print("│   └── clean/")
    print("├── evaluation/")
    print("│   ├── noisy/")
    print("│   └── clean/")
    print("├── training_pairs.csv")
    print("└── evaluation_pairs.csv")

if __name__ == "__main__":
    main()
