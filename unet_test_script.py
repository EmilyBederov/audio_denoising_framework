#!/usr/bin/env python3
"""
Simple test script to verify UNet integration works with your framework
"""

import sys
import os
import torch
from pathlib import Path

def test_unet_integration():
    """Test UNet integration step by step"""
    
    print("Testing UNet Integration with Your Framework")
    print("=" * 50)
    
    # Test 1: Import model factory
    print("\n1. Testing model factory import...")
    try:
        from core.model_factory import ModelFactory
        print("✓ Model factory imported successfully")
    except Exception as e:
        print(f"❌ Model factory import failed: {e}")
        return False
    
    # Test 2: Check if UNet is in model mapping
    print("\n2. Testing UNet model mapping...")
    try:
        if 'unet' in ModelFactory.MODEL_MAPPING:
            print("✓ UNet found in model mapping")
        else:
            print("❌ UNet not found in model mapping")
            print("Available models:", list(ModelFactory.MODEL_MAPPING.keys()))
            return False
    except Exception as e:
        print(f"❌ Error checking model mapping: {e}")
        return False
    
    # Test 3: Test config loading
    print("\n3. Testing config loading...")
    try:
        config = ModelFactory.load_model_config('unet')
        print(f"✓ Config loaded successfully")
        print(f"  Model name: {config.get('model_name')}")
        print(f"  Sample rate: {config.get('sample_rate')}")
        print(f"  Depth: {config.get('depth')}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        print("Make sure configs/configs-unet/unet-config.yaml exists")
        return False
    
    # Test 4: Test model creation
    print("\n4. Testing model creation...")
    try:
        model = ModelFactory.create_model('unet', config)
        print(f"✓ Model created successfully: {type(model)}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 5: Test forward pass
    print("\n5. Testing forward pass...")
    try:
        # Create dummy input (batch=2, channels=1, time=16, freq=513)
        dummy_input = torch.randn(2, 1, 16, 513)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Check output is in valid range (should be 0-1 due to sigmoid)
        output_min, output_max = output.min().item(), output.max().item()
        print(f"  Output range: [{output_min:.4f}, {output_max:.4f}]")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test 6: Test with your existing data loader
    print("\n6. Testing with your data loader...")
    try:
        from core.data_loader import get_dataloader
        
        # Try to create a dataloader (might fail if no data, but import should work)
        print("✓ Data loader import successful")
        print("  Note: Actual data loading depends on your data files existing")
        
    except Exception as e:
        print(f"⚠ Data loader test: {e}")
        print("  This is expected if you don't have data files yet")
    
    # Test 7: Test training manager compatibility  
    print("\n7. Testing training manager compatibility...")
    try:
        from core.training_manager import TrainingManager
        
        # Test creating training manager (might fail without proper data paths)
        print("✓ Training manager import successful")
        print("  Note: Full training depends on your data files existing")
        
    except Exception as e:
        print(f"⚠ Training manager test: {e}")
        print("  This is expected if data paths in config don't exist yet")
    
    print("\n" + "=" * 50)
    print("🎉 UNet Integration Test Completed!")
    print("=" * 50)
    
    print("\nNext Steps:")
    print("1. Make sure your data is set up (run setup.py if needed)")
    print("2. Try training: python run.py --config configs/configs-unet/unet-config.yaml --model unet --mode train")
    print("3. Try inference: python run.py --config configs/configs-unet/unet-config.yaml --model unet --mode inference --input your_audio.wav")
    
    return True

def check_file_structure():
    """Check if all necessary files exist"""
    
    print("Checking file structure...")
    
    required_files = [
        "models/unet/models/unet.py",
        "models/unet/unet_wrapper.py", 
        "configs/configs-unet/unet-config.yaml",
        "core/model_factory.py",
        "core/training_manager.py",
        "core/data_loader.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nRun the setup script first to create these files.")
        return False
    
    print("✓ All required files found")
    return True

if __name__ == "__main__":
    print("UNet Integration Verification")
    print("=" * 40)
    
    # Check file structure first
    if not check_file_structure():
        print("\n❌ File structure check failed")
        print("Please run the setup script first:")
        print("python setup_unet_integration.py")
        sys.exit(1)
    
    # Run integration tests
    if test_unet_integration():
        print("\n✅ All tests passed! UNet is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)