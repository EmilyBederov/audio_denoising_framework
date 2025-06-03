import time
import os
import torch
import torchaudio
import torchaudio.transforms as T
import yaml
from pathlib import Path
import sys

# Add your project to path
sys.path.append(str(Path(__file__).parent))
from core.model_factory import ModelFactory

# --- CONFIG ---
wav_path = "data/noisy/p234_007.wav"  # Change this to any noisy audio file you have
output_path = "data/denoised/p234_007.wav"
model_path = "outputs/cleanunet2/cleanunet2_best.pth"  # YOUR TRAINED MODEL!
config_path = "configs/configs-cleanunet2/cleanunet2-config.yaml"
sample_rate = 16000

print("=== TESTING YOUR TRAINED CLEANUNET2 MODEL ===")

# --- Load model using your training framework ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load config (same as training)
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Create model using your framework (ensures consistency)
model = ModelFactory.create_model('cleanunet2', config)
model.to(device)

# Load your trained weights
if os.path.exists(model_path):
    print(f"Loading trained model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Debug: Check what's in the checkpoint
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Training loss: {checkpoint.get('loss', 'unknown'):.6f}")
    elif 'state_dict' in checkpoint:
        # Add 'model.' prefix to match wrapper structure
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = f"model.{key}"
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        print(f"Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Training loss: {checkpoint.get('loss', 'unknown'):.6f}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print("Loaded direct state dict")
else:
    print(f"ERROR: Model not found at {model_path}")
    print("Available models:")
    model_dir = Path(model_path).parent
    if model_dir.exists():
        for f in model_dir.glob("*.pth"):
            print(f"  {f}")
    exit(1)

model.eval()

# --- Load audio ---
if not os.path.exists(wav_path):
    print(f"ERROR: Audio file not found: {wav_path}")
    print("Please provide a valid noisy audio file.")
    print("You can use any .wav file or pick one from your validation data.")
    exit(1)

print(f"Loading audio: {wav_path}")
waveform, sr = torchaudio.load(wav_path)
original_duration = waveform.shape[-1] / sr
print(f"Original audio: {waveform.shape}, {sr} Hz, {original_duration:.2f} sec")

if sr != sample_rate:
    print(f"Resampling from {sr} Hz to {sample_rate} Hz")
    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

if waveform.shape[0] > 1:  # Convert stereo to mono
    print("Converting stereo to mono")
    waveform = waveform.mean(dim=0, keepdim=True)

waveform = waveform.unsqueeze(0).to(device)  # [1, 1, T]
duration = waveform.shape[-1] / sample_rate

# --- Compute spectrogram using SAME parameters as training ---
spec_transform = T.Spectrogram(
    n_fft=config.get('n_fft', 1024),
    hop_length=config.get('hop_length', 256),
    win_length=config.get('win_length', 1024),
    power=config.get('power', 1.0),  # Match training exactly
    normalized=True,
    center=False,  # Match training configuration
    return_complex=False
).to(device)

spectrogram = spec_transform(waveform.squeeze(0))  # [F, T]
print(f"Spectrogram shape: {spectrogram.shape}")
spectrogram = spectrogram.unsqueeze(0)  # [1, F, T]

# --- Warm-up ---
print("Warming up model...")
with torch.no_grad():
    try:
        _ = model(waveform, spectrogram)
        print("✓ Warm-up successful")
    except Exception as e:
        print(f"✗ Warm-up failed: {e}")
        exit(1)

# --- Timed inference ---
print("Starting inference...")
start = time.time()
with torch.no_grad():
    output = model(waveform, spectrogram)
    # Handle tuple output (enhanced_audio, enhanced_spec)
    if isinstance(output, tuple):
        print(f"Model returned a tuple with {len(output)} elements")
        denoised = output[0]  # First element is the enhanced audio
    else:
        denoised = output

if device.type == "cuda":
    torch.cuda.synchronize()
end = time.time()

# --- Save output ---
print(f"Denoised output shape: {denoised.shape}")
print(f"Denoised range: [{denoised.min():.4f}, {denoised.max():.4f}]")

# Audio should already be clipped to [-1, 1] by your wrapper
denoised_cpu = denoised.squeeze().cpu()
original_cpu = waveform.squeeze().cpu()

# Create output directory
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save both original and denoised for comparison
torchaudio.save(output_path, denoised_cpu.unsqueeze(0), sample_rate)
original_path = output_path.replace('.wav', '_original.wav')
torchaudio.save(original_path, original_cpu.unsqueeze(0), sample_rate)

# --- RTF calculation ---
elapsed = end - start
rtf = elapsed / duration

print(f"\n=== RESULTS ===")
print(f"✓ Original saved: {original_path}")
print(f"✓ Denoised saved: {output_path}")
print(f"⏱  Inference time: {elapsed:.4f} sec")
print(f"🎵 Audio duration: {duration:.2f} sec")
print(f"🚀 Real-Time Factor (RTF): {rtf:.4f}")

if rtf < 1.0:
    print(f"✅ Real-time capable! ({1/rtf:.1f}x faster than real-time)")
else:
    print(f"❌ Too slow for real-time ({rtf:.1f}x slower than real-time)")

print(f"\n=== NEXT STEPS ===")
print(f"1. Download both files to listen:")
print(f"   scp emilybederov@dgx-master:{os.path.abspath(original_path)} ./")
print(f"   scp emilybederov@dgx-master:{os.path.abspath(output_path)} ./")
print(f"2. Compare the original (noisy) vs denoised audio!")
print(f"3. Your CleanUNet2 model is working! 🎉")
