#!/usr/bin/env python3
"""
Real-time Audio Denoising Test Script using Original CleanUNet2 Implementation
"""
import time
import os
import sys
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

try:
    import sounddevice as sd
    print("INFO: Audio playback available")
except (ImportError, OSError) as e:
    print("INFO: Audio playback disabled (server environment)")

# FIXED: Import from your existing models directory
sys.path.insert(0, str(Path(__file__).parent))
from models.cleanunet2.models.cleanunet2 import CleanUNet2  # Your existing file!

def load_original_model_with_checkpoint(checkpoint_path, device, use_defaults=True):
    """
    Load the original CleanUNet2 model with checkpoint handling
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        device: Device to load the model on
        use_defaults: If True, use original paper defaults. If False, load from config.
    """
    print(f"Loading original CleanUNet2 model...")
    
    if use_defaults:
        print("✅ Using original paper defaults")
        # Use original defaults (paper configuration) - NO PARAMETERS NEEDED!
        model = CleanUNet2().to(device)
    else:
        # Custom configuration (if you want to override defaults)
        model = CleanUNet2(
            # Only specify if you want to override the perfect defaults
            # cleanunet_channels_H=64,      # Already default
            # cleanunet_max_H=768,          # Already default
            # etc...
        ).to(device)
    
    print(f"Model created with original defaults")
    
    # Load checkpoint with prefix handling
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats and strip 'model.' prefix if present
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Strip 'model.' prefix from keys if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load state dict with error handling
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("✅ Model loaded successfully with strict=True")
        except RuntimeError as e:
            print(f"WARNING: Strict loading failed: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ Model loaded with strict=False (some parameters may be missing)")
            
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        raise
        
    model.eval()
    return model

def run_inference_test():
    """
    Run audio denoising inference test using original CleanUNet2
    """
    print("🎵 Original CleanUNet2 Audio Denoising Test")
    print("=" * 60)
    
    # --- Configuration ---
    MODEL_PATH = "outputs/cleanunet2/cleanunet2_best.pth"
    TEST_AUDIO_DIR = "data/testdata/noisy"
    OUTPUT_DIR = "data/testdata/enhanced_original"
    SAMPLE_RATE = 16000
    
    # Audio processing parameters (matching original implementation)
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    POWER = 1.0
    
    print(f"Model path: {MODEL_PATH}")
    print(f"Test audio directory: {TEST_AUDIO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Count test files
    test_files = []
    if os.path.exists(TEST_AUDIO_DIR):
        test_files = [f for f in os.listdir(TEST_AUDIO_DIR) if f.endswith(('.wav', '.flac', '.mp3'))]
    
    print(f"Number of samples: {len(test_files)}")
    print("=" * 60)
    
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Model Setup ---
    # Load the ORIGINAL CleanUNet2 with paper defaults
    try:
        model = load_original_model_with_checkpoint(MODEL_PATH, device, use_defaults=True)
        print("✅ Original CleanUNet2 loaded and set to evaluation mode")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return
    
    # --- Audio Processing Setup ---
    # Spectrogram transform (matching original implementation)
    spec_transform = T.Spectrogram(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        power=POWER,
        normalized=True,
        center=False  # Match original implementation
    ).to(device)
    
    print(f"Spectrogram config: n_fft={N_FFT}, hop_length={HOP_LENGTH}, power={POWER}")
    
    # --- Test Files ---
    if test_files:
        print(f"\n🎵 Processing {len(test_files)} test files...")
        
        total_time = 0
        total_duration = 0
        
        for i, filename in enumerate(test_files):
            input_path = os.path.join(TEST_AUDIO_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, f"enhanced_{filename}")
            
            try:
                elapsed, duration = process_audio_file(
                    input_path, output_path, model, spec_transform, 
                    device, SAMPLE_RATE
                )
                
                total_time += elapsed
                total_duration += duration
                
                rtf = elapsed / duration if duration > 0 else float('inf')
                print(f"  [{i+1:2d}/{len(test_files)}] {filename}: "
                      f"{elapsed:.3f}s / {duration:.2f}s (RTF: {rtf:.3f})")
                      
            except Exception as e:
                print(f"  [{i+1:2d}/{len(test_files)}] {filename}: ERROR - {e}")
        
        # Summary
        if total_duration > 0:
            avg_rtf = total_time / total_duration
            print(f"\n📊 Summary:")
            print(f"  Total inference time: {total_time:.3f}s")
            print(f"  Total audio duration: {total_duration:.2f}s") 
            print(f"  Average RTF: {avg_rtf:.4f}")
            print(f"  Performance: {'✅ Real-time capable' if avg_rtf < 1.0 else '❌ Too slow for real-time'}")
    
    else:
        # Test with synthetic audio
        print("No test files found. Creating synthetic test...")
        
        duration_sec = 3.0
        t = torch.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec))
        clean_signal = torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz tone
        noise = torch.randn_like(clean_signal) * 0.1
        noisy_signal = clean_signal + noise
        
        # Save synthetic test file
        test_input = os.path.join(OUTPUT_DIR, "synthetic_noisy.wav")
        torchaudio.save(test_input, noisy_signal.unsqueeze(0), SAMPLE_RATE)
        
        # Process synthetic file
        test_output = os.path.join(OUTPUT_DIR, "synthetic_enhanced.wav")
        elapsed, duration = process_audio_file(
            test_input, test_output, model, spec_transform, device, SAMPLE_RATE
        )
        
        rtf = elapsed / duration
        print(f"Synthetic test: {elapsed:.3f}s / {duration:.2f}s (RTF: {rtf:.3f})")
        print(f"Performance: {'✅ Real-time capable' if rtf < 1.0 else '❌ Too slow for real-time'}")

def process_audio_file(input_path, output_path, model, spec_transform, device, sample_rate):
    """
    Process a single audio file through the original denoising model
    """
    # Load audio
    waveform, sr = torchaudio.load(input_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Calculate duration
    duration = waveform.shape[-1] / sample_rate
    
    # Move to device and add batch dimension
    waveform = waveform.unsqueeze(0).to(device)  # [1, 1, T]
    
    # Compute spectrogram (matching original implementation)
    spectrogram = spec_transform(waveform.squeeze(0))  # [F, T]
    spectrogram = spectrogram.unsqueeze(0).to(device)  # [1, F, T]
    
    # Run inference with timing
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    
    with torch.no_grad():
        # Original CleanUNet2 returns (enhanced_audio, enhanced_spec)
        enhanced_audio, enhanced_spec = model(waveform, spectrogram)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()
    
    # Save enhanced audio
    enhanced_audio = enhanced_audio.squeeze().cpu().clamp(-1, 1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, enhanced_audio.unsqueeze(0), sample_rate)
    
    elapsed = end_time - start_time
    return elapsed, duration

def main():
    """
    Main function with error handling
    """
    try:
        run_inference_test()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
