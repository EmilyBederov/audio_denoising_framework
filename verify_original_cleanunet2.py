#!/usr/bin/env python3
"""
Comprehensive verification script to ensure Original CleanUNet2 is working in your framework
"""
import sys
import torch
import yaml
import os
from pathlib import Path

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing Critical Imports...")
    
    try:
        # Test base imports
        from models.cleanunet2.models.cleanunet import CleanUNet
        print("✅ CleanUNet (base) import successful")
        
        from models.cleanunet2.models.cleanspecnet import CleanSpecNet  
        print("✅ CleanSpecNet import successful")
        
        from models.cleanunet2.models.cleanunet2 import CleanUNet2
        print("✅ CleanUNet2 (hybrid) import successful")
        
        # Test framework imports
        from core.model_factory import ModelFactory
        print("✅ ModelFactory import successful")
        
        from models.cleanunet2.cleanunet2_wrapper import CleanUNet2Wrapper
        print("✅ CleanUNet2Wrapper import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_original_model_creation():
    """Test creating model with original defaults"""
    print("\n🏗️  Testing Original Model Creation...")
    
    try:
        from models.cleanunet2.models.cleanunet2 import CleanUNet2
        
        # Test 1: Create with no parameters (should use original defaults)
        model = CleanUNet2()
        print("✅ CleanUNet2 created with original defaults")
        
        # Test 2: Count parameters  
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Total parameters: {total_params:,}")
        
        # Verify this matches original paper scale (10-15M parameters)
        if total_params > 8_000_000:
            print("✅ Parameter count looks correct for original configuration")
        else:
            print("⚠️  Parameter count seems low - may not be using original defaults")
            
        # Test 3: Check specific architecture components
        print(f"   CleanUNet channels_H: {model.clean_unet.channels_H}")
        print(f"   CleanUNet max_H: {model.clean_unet.max_H}")
        print(f"   CleanUNet encoder_n_layers: {model.clean_unet.encoder_n_layers}")
        print(f"   CleanSpecNet hidden_dim: {model.clean_spec_net.tsfm_projection.out_features}")
        
        # Verify original defaults are being used
        expected_values = {
            'channels_H': 64,
            'max_H': 768, 
            'encoder_n_layers': 8
        }
        
        all_correct = True
        for param, expected in expected_values.items():
            actual = getattr(model.clean_unet, param)
            if actual == expected:
                print(f"✅ {param}: {actual} (correct)")
            else:
                print(f"❌ {param}: {actual}, expected {expected}")
                all_correct = False
        
        if all_correct:
            print("✅ All architecture parameters match original defaults")
        else:
            print("⚠️  Some parameters don't match original defaults")
            
        return True, model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_framework_integration():
    """Test framework integration with original model"""
    print("\n🔗 Testing Framework Integration...")
    
    try:
        from core.model_factory import ModelFactory
        
        # Test loading config
        config_path = "configs/configs-cleanunet2/cleanunet2-config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        print(f"✅ Config loaded from: {config_path}")
        
        # Check use_original_defaults
        use_original = config.get("network_config", {}).get("use_original_defaults", False)
        if use_original:
            print("✅ Config set to use original defaults")
        else:
            print("⚠️  Config NOT set to use original defaults")
            
        # Test model creation through factory
        model = ModelFactory.create_model("cleanunet2", config)
        print("✅ Model created through ModelFactory")
        
        # Check parameter count again
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters via factory: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\n🔄 Testing Forward Pass...")
    
    try:
        from models.cleanunet2.models.cleanunet2 import CleanUNet2
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")
        
        # Create model
        model = CleanUNet2().to(device)
        model.eval()
        
        # Create realistic dummy inputs
        batch_size = 2
        audio_length = 16000 * 3  # 3 seconds at 16kHz
        freq_bins = 513
        time_frames = (audio_length // 256) + 1  # Based on hop_length=256
        
        noisy_waveform = torch.randn(batch_size, 1, audio_length).to(device)
        noisy_spectrogram = torch.randn(batch_size, freq_bins, time_frames).to(device)
        
        print(f"   Input shapes: audio={noisy_waveform.shape}, spec={noisy_spectrogram.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(noisy_waveform, noisy_spectrogram)
            
        # Check outputs
        if isinstance(outputs, tuple):
            denoised_audio, denoised_spec = outputs
            print(f"   Output shapes: audio={denoised_audio.shape}, spec={denoised_spec.shape}")
            
            # Verify shapes
            if denoised_audio.shape == noisy_waveform.shape:
                print("✅ Audio output shape correct")
            else:
                print(f"❌ Audio shape mismatch")
                
            if denoised_spec.shape == noisy_spectrogram.shape:
                print("✅ Spectrogram output shape correct")
            else:
                print(f"❌ Spectrogram shape mismatch")
                
        else:
            print(f"❌ Expected tuple output, got {type(outputs)}")
            return False
            
        print("✅ Forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_paths():
    """Test that data paths in config are valid"""
    print("\n📁 Testing Config Data Paths...")
    
    try:
        config_path = "configs/configs-cleanunet2/cleanunet2-config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        train_csv = config.get("trainset", {}).get("csv_path")
        val_csv = config.get("valset", {}).get("csv_path")
        
        if train_csv:
            if os.path.exists(train_csv):
                print(f"✅ Training CSV found: {train_csv}")
            else:
                print(f"⚠️  Training CSV not found: {train_csv}")
                print(f"   You need to create this file with your training data")
        else:
            print("❌ No training CSV path in config")
            
        if val_csv:
            if os.path.exists(val_csv):
                print(f"✅ Validation CSV found: {val_csv}")
            else:
                print(f"⚠️  Validation CSV not found: {val_csv}")
                print(f"   You need to create this file with your validation data")
        else:
            print("❌ No validation CSV path in config")
            
        return True
        
    except Exception as e:
        print(f"❌ Config path test failed: {e}")
        return False

def show_next_steps():
    """Show what to do next"""
    print("\n" + "="*60)
    print("📋 NEXT STEPS")
    print("="*60)
    print("1. Create your training data CSV files:")
    print("   - data/training_pairs_16khz.csv")
    print("   - data/evaluation_pairs_16khz.csv")
    print()
    print("2. CSV format should be:")
    print("   noisy_path,clean_path")
    print("   path/to/noisy1.wav,path/to/clean1.wav")
    print("   path/to/noisy2.wav,path/to/clean2.wav")
    print()
    print("3. Run training with:")
    print("   python run.py --config configs/configs-cleanunet2/cleanunet2-config.yaml --model cleanunet2 --mode train")
    print()
    print("4. For inference:")
    print("   python run.py --config configs/configs-cleanunet2/cleanunet2-config.yaml --model cleanunet2 --mode inference --input audio.wav")

def main():
    """Run all verification tests"""
    print("🧪 Original CleanUNet2 Framework Verification")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Model Creation
    success, model = test_original_model_creation()
    if success:
        tests_passed += 1
    
    # Test 3: Framework Integration  
    if test_framework_integration():
        tests_passed += 1
        
    # Test 4: Forward Pass
    if test_forward_pass():
        tests_passed += 1
        
    # Test 5: Config Paths
    if test_config_paths():
        tests_passed += 1
    
    # Results
    print("\n" + "="*60)
    print(f"📊 VERIFICATION RESULTS: {tests_passed}/{total_tests} tests passed")
    print("="*60)
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Original CleanUNet2 is ready to use.")
        show_next_steps()
    elif tests_passed >= 3:
        print("⚠️  Most tests passed. Check the failed tests above.")
        show_next_steps()
    else:
        print("❌ Many tests failed. Review the fixes needed above.")
        print("\nKey files to check:")
        print("- models/cleanunet2/models/cleanunet.py (fix imports)")
        print("- models/cleanunet2/models/__init__.py (create if missing)")
        print("- configs/configs-cleanunet2/cleanunet2-config.yaml (check paths)")

if __name__ == "__main__":
    main()
