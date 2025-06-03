#!/usr/bin/env python3
"""
Quick test to see what's wrong with audio loading
"""

import torchaudio
import torch

def test_audio_file(file_path):
    print(f"Testing: {file_path}")
    
    try:
        # Check what backends are available
        print(f"Available backends: {torchaudio.list_audio_backends()}")
        
        # Try to load the file
        audio, sr = torchaudio.load(file_path)
        print(f" Success! Shape: {audio.shape}, Sample rate: {sr}")
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        return False

def test_backend_info():
    print("Torchaudio backend info:")
    print(f"Available backends: {torchaudio.list_audio_backends()}")
    print(f"Current backend: {torchaudio.get_audio_backend()}")

if __name__ == "__main__":
    test_backend_info()
    print("\n" + "="*50)
    
    # Test the problematic file
    test_audio_file("data/training/noisy/sample2207.wav")
    test_audio_file("data/training/clean/sample2207.wav")
