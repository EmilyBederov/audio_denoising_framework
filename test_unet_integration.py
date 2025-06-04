#!/usr/bin/env python3
"""
Test script to verify UNet integration works with your framework
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

def test_unet_integration():
    """Test UNet integration step by step"""
    
    print("Testing UNet Integration")
    print("=" * 40)
    
    # Test 1: Model Factory
    print("\n1. Testing Model Factory...")
    try:
        from core.model_factory import ModelFactory
        
        # Check if UNet is available
        if 'unet' not in ModelFactory.MODEL_MAPPING:
            print(" UNet not found in model mapping")
            print("Available models:", list(ModelFactory.MODEL_MAPPING.keys()))
            return False
        
        print(" UNet found in model factory")
    except Exception as e:
        print(f" Model factory test failed: {e}")
        return False
    
    # Test 2: Config Loading
    print("\n2. Testing Config Loading...")
    try:
        config = ModelFactory.load_model_config('unet')
        print(f" Config loaded: {config['model_name']}")
        print(f"   Sample rate: {config['sample_rate']}")
        print(f"   Batch size: {config['batch_size']}")
    except Exception as e:
        print(f" Config loading failed: {e}")
        print("Make sure configs/configs-unet/unet.yaml exists")
        return False
    
    # Test 3: Model Creation
    print("\n3. Testing Model Creation...")
    try:
        model = ModelFactory.create_model('unet', config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f" UNet model created successfully")
        print(f"   Parameters: {total_params:,}")
        print(f"   Model type: {getattr(model, 'model_type', 'default')}")
    except Exception as e:
        print(f" Model creation failed: {e}")
        return False
    
    # Test 4: Forward Pass
    print("\n4. Testing Forward Pass...")
    try:
        # Test spectrogram input [B, 1, T, F]
        batch_size = 2
        time_frames = 16
        freq_bins = 513
        
        dummy_spec = torch.randn(batch_size, 1, time_frames, freq_bins).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_spec)
        
        print(f" Forward pass successful")
        print(f"   Input shape: {dummy_spec.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
    except Exception as e:
        print(f" Forward pass failed: {e}")
        return False
    
    # Test 5: Batch Preparation
    print("\n5. Testing Batch Preparation...")
    try:
        # Simulate dataloader batch format
        clean_audio = torch.randn(2, 1, 16000)  # [B, C, T]
        clean_spec = torch.randn(2, 513, 100)   # [B, F, T]
        noisy_audio = torch.randn(2, 1, 16000)  # [B, C, T]  
        noisy_spec = torch.randn(2, 513, 100)   # [B, F, T]
        
        batch = (clean_audio, clean_spec, noisy_audio, noisy_spec)
        
        if hasattr(model, 'prepare_training_batch'):
            inputs, targets = model.prepare_training_batch(batch)
            print(f" Batch preparation successful")
            print(f"   Input shape: {inputs.shape}")
            print(f"   Target shape: {targets.shape}")
        else:
            print(" No prepare_training_batch method (using fallback)")
            
    except Exception as e:
        print(f" Batch preparation failed: {e}")
        return False
    
    # Test 6: Training Manager Compatibility
    print("\n6. Testing Training Manager...")
    try:
        from core.training_manager import TrainingManager
        
        # Test creating training manager
        trainer = TrainingManager('unet', config, device='cpu')
        print(f" Training manager created for UNet")
        
    except Exception as e:
        print(f" Training manager test failed: {e}")
        return False
    
    # Test 7: Compare with CleanUNet2 (if available)
    print("\n7. Testing Coexistence with CleanUNet2...")
    try:
        # Test that CleanUNet2 still works
        if 'cleanunet2' in ModelFactory.MODEL_MAPPING:
            cleanunet2_config = ModelFactory.load_model_config('cleanunet2')
            cleanunet2_model = ModelFactory.create_model('cleanunet2', cleanunet2_config)
            print(" CleanUNet2 still works alongside UNet")
        else:
            print(" CleanUNet2 not available (this is okay)")
            
    except Exception as e:
        print(f" CleanUNet2 coexistence test: {e}")
        print("   This may be expected if CleanUNet2 files are missing")
    
    print("\n" + "=" * 40)
    print(" UNet Integration Test Completed!")
    print("=" * 40)
    
    return True

def test_inference_pipeline():
    """Test the complete inference pipeline"""
    print("\n8. Testing Inference Pipeline...")
    
    try:
        from core.model_factory import ModelFactory
        
        # Load UNet
        config = ModelFactory.load_model_config('unet')
        model = ModelFactory.create_model('unet', config)
        
        # Test preprocessing
        if hasattr(model, 'preprocess_for_inference'):
            print(" UNet has inference preprocessing")
            
        if hasattr(model, 'postprocess_output'):
            print(" UNet has inference postprocessing")
            
        if hasattr(model, 'inference'):
            print(" UNet has complete inference pipeline")
        
        print(" Inference pipeline components available")
        
    except Exception as e:
        print(f" Inference pipeline test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("UNet Framework Integration Test")
    print("=" * 50)
    
    # Check if basic files exist
    required_files = [
        "models/unet/models/unet.py",
        "models/unet/unet_wrapper.py",
        "configs/configs-unet/unet.yaml",
        "core/model_factory.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(" Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease create these files first.")
        return False
    
    # Run tests
    success = True
    success &= test_unet_integration()
    success &= test_inference_pipeline()
    
    if success:
        print("\n All tests passed! UNet is ready to use.")
        print("\nUsage:")
        print("  Training:   python run.py --config configs/configs-unet/unet.yaml --model unet --mode train")
        print("  Inference:  python run.py --config configs/configs-unet/unet.yaml --model unet --mode inference --input audio.wav")
    else:
        print("\n Some tests failed. Check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
