# Create check_fixed.py
import torch
import yaml
from core.model_factory import ModelFactory

# Load config
with open('configs/configs-cleanunet2/cleanunet2-config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = ModelFactory.create_model('cleanunet2', config).cuda()

# Use consistent dimensions - match your dataloader settings
sample_rate = 16000
n_fft = 1024
hop_length = 256
win_length = 1024

# Create audio that gives consistent spectrogram size
audio_length = 16000  # 1 second
dummy_audio = torch.randn(2, 1, audio_length).cuda()

# Compute spectrogram with same settings as dataloader
stft = torch.stft(
    dummy_audio.squeeze(1), 
    n_fft=n_fft, 
    hop_length=hop_length,
    win_length=win_length, 
    return_complex=True,
    center=False,
    normalized=True
)
dummy_spec = torch.abs(stft)  # [B, F, T]

print(f"Audio shape: {dummy_audio.shape}")
print(f"Spec shape: {dummy_spec.shape}")

model.eval()
with torch.no_grad():
    try:
        output = model(dummy_audio, dummy_spec)
        print(f'✓ Model works! Output audio range: {output[0].min():.4f} to {output[0].max():.4f}')
        print(f'Output mean: {output[0].mean():.4f}')
        print(f'Is all zeros? {torch.allclose(output[0], torch.zeros_like(output[0]))}')
    except Exception as e:
        print(f'✗ Model failed: {e}')
