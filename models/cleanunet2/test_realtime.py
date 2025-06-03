import time
import os
import torch
import torchaudio
import torchaudio.transforms as T
from models import CleanUNet2

# --- CONFIG ---
wav_path = "data/noisy/p234_007.wav"
output_path = "data/denoised/p234_007.wav"
model_path = "output/cleanunet2_5sample.pth"
sample_rate = 16000

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CleanUNet2(
    cleanunet_input_channels=1,
    cleanunet_output_channels=1,
    cleanunet_channels_H=32,
    cleanunet_max_H=256,
    cleanunet_encoder_n_layers=5,
    cleanunet_kernel_size=4,
    cleanunet_stride=2,
    cleanunet_tsfm_n_layers=2,
    cleanunet_tsfm_n_head=4,
    cleanunet_tsfm_d_model=128,
    cleanunet_tsfm_d_inner=512,
    cleanspecnet_input_channels=513,  # Match the trained model configuration
    cleanspecnet_num_conv_layers=3,
    cleanspecnet_kernel_size=3,
    cleanspecnet_stride=1,
    cleanspecnet_num_attention_layers=2,
    cleanspecnet_num_heads=4,
    cleanspecnet_hidden_dim=128,
    cleanspecnet_dropout=0.1
).to(device)

# Load state with safer option
state = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
model.eval()

# --- Load audio ---
waveform, sr = torchaudio.load(wav_path)
if sr != sample_rate:
    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

if waveform.shape[0] > 1:  # Convert stereo to mono
    waveform = waveform.mean(dim=0, keepdim=True)

waveform = waveform.unsqueeze(0).to(device)  # [1, 1, T]
duration = waveform.shape[-1] / sample_rate

# --- Compute spectrogram using the same parameters as during training ---
spec_transform = T.Spectrogram(
    n_fft=1024,  # This gives 513 frequency bins (n_fft//2 + 1)
    hop_length=256,
    win_length=1024,
    power=2,
    normalized=True,
    center=False,  # Match the training configuration
    return_complex=False
).to(device)

spectrogram = spec_transform(waveform.squeeze(0))  # [F, T]
print(f"Spectrogram shape: {spectrogram.shape}")  # Should be [513, T]
spectrogram = spectrogram.unsqueeze(0).to(device)  # [1, F, T]

# --- Warm-up ---
print("Warming up model...")
with torch.no_grad():
    _ = model(waveform, spectrogram)
print("Warm-up complete")

# --- Timed inference ---
print("Starting inference...")
start = time.time()
with torch.no_grad():
    output = model(waveform, spectrogram)
    # Handle if output is a tuple
    if isinstance(output, tuple):
        print(f"Model returned a tuple with {len(output)} elements")
        denoised = output[0]  # First element is the waveform
    else:
        denoised = output
torch.cuda.synchronize() if device.type == "cuda" else None
end = time.time()

# --- Save output ---
print(f"Denoised output shape: {denoised.shape}")
denoised = denoised.squeeze().cpu().clamp(-1, 1)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torchaudio.save(output_path, denoised.unsqueeze(0), sample_rate)

# --- RTF calculation ---
elapsed = end - start
rtf = elapsed / duration
print(f" Saved to: {output_path}")
print(f" Inference time: {elapsed:.4f} sec")
print(f" Audio duration: {duration:.2f} sec")
print(f" Real-Time Factor (RTF): {rtf:.4f} → {' Real-time' if rtf < 1 else ' Too slow'}")