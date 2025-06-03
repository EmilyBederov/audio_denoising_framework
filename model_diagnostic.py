#!/usr/bin/env python3
"""
Model Diagnostic Script - Check if model is actually learning to denoise
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add model path
current_dir = Path(__file__).parent
models_dir = current_dir / "models" / "cleanunet2"
sys.path.insert(0, str(models_dir))

from models.cleanunet2 import CleanUNet2

def load_model_with_prefix_handling(model, checkpoint_path, device):
    """Load model with prefix handling"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    # Strip 'model.' prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    return model

def analyze_model_behavior():
    """Analyze if the model is actually denoising"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model with your trained weights
    model = CleanUNet2(
        cleanunet_input_channels=1,
        cleanunet_output_channels=1,
        cleanunet_channels_H=48,
        cleanunet_max_H=512,
        cleanunet_encoder_n_layers=6,
        cleanunet_kernel_size=4,
        cleanunet_stride=2,
        cleanunet_tsfm_n_layers=3,
        cleanunet_tsfm_n_head=8,
        cleanunet_tsfm_d_model=256,
        cleanunet_tsfm_d_inner=1024,
        cleanspecnet_input_channels=513,
        cleanspecnet_num_conv_layers=3,
        cleanspecnet_kernel_size=3,
        cleanspecnet_stride=1,
        cleanspecnet_num_attention_layers=2,
        cleanspecnet_num_heads=4,
        cleanspecnet_hidden_dim=128,
        cleanspecnet_dropout=0.1
    ).to(device)
    
    # Load your trained weights
    model = load_model_with_prefix_handling(model, "outputs/cleanunet2/cleanunet2_best.pth", device)
    model.eval()
    
    # Create test signals
    sr = 16000
    duration = 3.0
    t = torch.linspace(0, duration, int(sr * duration))
    
    # Test 1: Pure sine wave (should pass through unchanged)
    clean_sine = torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz
    
    # Test 2: Sine wave + noise (should remove noise)
    noise = torch.randn_like(clean_sine) * 0.3
    noisy_sine = clean_sine + noise
    
    # Test 3: Pure noise (should be heavily attenuated)
    pure_noise = torch.randn_like(clean_sine) * 0.5
    
    print("🔬 Model Diagnostic Tests")
    print("=" * 50)
    
    # Test each signal
    test_signals = [
        ("Clean Sine Wave", clean_sine),
        ("Noisy Sine Wave", noisy_sine), 
        ("Pure Noise", pure_noise)
    ]
    
    spec_transform = T.Spectrogram(
        n_fft=1024, hop_length=256, win_length=1024, 
        power=1.0, normalized=True, center=False
    ).to(device)
    
    results = []
    
    for name, signal in test_signals:
        print(f"\n🧪 Testing: {name}")
        
        # Prepare input - FIXED: Move signal to device first
        signal_device = signal.to(device)
        waveform = signal_device.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        spectrogram = spec_transform(signal_device).unsqueeze(0)  # [1, F, T]
        
        # Run model
        with torch.no_grad():
            output = model(waveform, spectrogram)
            if isinstance(output, tuple):
                enhanced = output[0]
            else:
                enhanced = output
        
        # Convert back to numpy
        input_np = signal.cpu().numpy()
        output_np = enhanced.squeeze().cpu().numpy()
        
        # Calculate metrics
        input_rms = np.sqrt(np.mean(input_np**2))
        output_rms = np.sqrt(np.mean(output_np**2))
        
        # Measure similarity (correlation)
        correlation = np.corrcoef(input_np, output_np)[0, 1]
        
        # Measure attenuation
        attenuation_db = 20 * np.log10(output_rms / (input_rms + 1e-8))
        
        print(f"   Input RMS: {input_rms:.4f}")
        print(f"   Output RMS: {output_rms:.4f}")
        print(f"   Correlation: {correlation:.4f}")
        print(f"   Attenuation: {attenuation_db:.2f} dB")
        
        results.append({
            'name': name,
            'input': input_np,
            'output': output_np,
            'correlation': correlation,
            'attenuation_db': attenuation_db
        })
    
    # Analysis
    print(f"\n📊 Analysis:")
    print("=" * 50)
    
    noisy_result = results[1]  # Noisy sine wave
    noise_result = results[2]   # Pure noise
    
    # Check if model is working
    if noisy_result['correlation'] > 0.95:
        print("❌ PROBLEM: Model output almost identical to input!")
        print("   → Model is NOT learning to denoise")
        
    elif noise_result['attenuation_db'] > -6:
        print("❌ PROBLEM: Model doesn't attenuate noise enough!")
        print(f"   → Noise attenuation only {noise_result['attenuation_db']:.1f} dB")
        
    elif noisy_result['correlation'] < 0.7:
        print("❌ PROBLEM: Model distorts clean signal too much!")
        print("   → Model is over-processing")
        
    else:
        print("✅ Model seems to be working!")
        print(f"   → Noise attenuated: {noise_result['attenuation_db']:.1f} dB")
        print(f"   → Signal preserved: {noisy_result['correlation']:.3f} correlation")
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    for i, result in enumerate(results):
        # Waveform
        time = np.arange(len(result['input'])) / sr
        axes[i, 0].plot(time, result['input'], label='Input', alpha=0.7)
        axes[i, 0].plot(time, result['output'], label='Output', alpha=0.7)
        axes[i, 0].set_title(f"{result['name']} - Waveform")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Spectrum
        fft_input = np.abs(np.fft.fft(result['input']))[:len(result['input'])//2]
        fft_output = np.abs(np.fft.fft(result['output']))[:len(result['output'])//2]
        freqs = np.fft.fftfreq(len(result['input']), 1/sr)[:len(result['input'])//2]
        
        axes[i, 1].plot(freqs, 20*np.log10(fft_input + 1e-8), label='Input', alpha=0.7)
        axes[i, 1].plot(freqs, 20*np.log10(fft_output + 1e-8), label='Output', alpha=0.7)
        axes[i, 1].set_title(f"{result['name']} - Spectrum")
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Magnitude (dB)')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlim(0, 8000)  # Show up to 8kHz
    
    plt.tight_layout()
    plt.savefig('model_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Diagnostic plot saved as: model_diagnostic.png")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print("=" * 50)
    
    if noisy_result['correlation'] > 0.95:
        print("🔧 Try these fixes:")
        print("   1. Increase model capacity (more layers/channels)")
        print("   2. Train for more epochs (100+ instead of 25)")
        print("   3. Lower learning rate (0.0001 instead of 0.0002)")
        print("   4. Check training data quality")
        print("   5. Verify loss function is working")
    
    return results

if __name__ == "__main__":
    try:
        analyze_model_behavior()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
