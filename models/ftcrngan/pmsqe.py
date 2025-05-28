import torch
import torch.nn as nn
import torch.nn.functional as F

class PMSQE(nn.Module):
    def __init__(self, eps=1e-8, n_fft=512, hop_length=256, win_length=512):
        super(PMSQE, self).__init__()
        self.eps = eps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(self.win_length)

    def forward(self, enhanced_wave, clean_wave, pad_mask=None):
        """
        Args:
            enhanced_wave: Tensor [B, 1, T]
            clean_wave: Tensor [B, 1, T]
            pad_mask: Tensor [B, T] or None
        Returns:
            PMSQE loss (scalar), dummy aux term (for compatibility)
        """

        B, _, T = enhanced_wave.shape
        device = enhanced_wave.device
        window = self.window.to(device)

        # STFT: [B, F, T']
        enhanced_stft = torch.stft(enhanced_wave.squeeze(1), n_fft=self.n_fft,
                                   hop_length=self.hop_length,
                                   win_length=self.win_length,
                                   window=window, return_complex=True)
        clean_stft = torch.stft(clean_wave.squeeze(1), n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                win_length=self.win_length,
                                window=window, return_complex=True)

        enhanced_mag = enhanced_stft.abs() + self.eps
        clean_mag = clean_stft.abs() + self.eps

        # Log-spectral distortion
        log_ratio = torch.log10(enhanced_mag / clean_mag)
        loss = log_ratio.pow(2)

        if pad_mask is not None:
            # Downsample pad_mask to match time frames
            ratio = T // loss.shape[-1]
            pad_mask = pad_mask[:, ::ratio].unsqueeze(1)  # [B, 1, T']
            loss = loss * pad_mask

        loss = loss.mean()
        return loss, torch.tensor(0.0).to(loss.device)  # dummy aux term