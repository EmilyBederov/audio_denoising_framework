# Delete ALL content and paste the clean script from above
import time
import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import random
import sys
from pathlib import Path

# Try to import sounddevice, but make it optional
try:
    import sounddevice as sd
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    print("WARNING: Audio playback not available (sounddevice not installed)")
    print("Will save audio files instead for manual listening")
    AUDIO_PLAYBACK_AVAILABLE = False

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models" / "cleanunet2"))

# Import CleanUNet2 
try:
    from models.cleanunet2.models.cleanunet2 import CleanUNet2
except ImportError:
    try:
        from models.cleanunet2 import CleanUNet2
    except ImportError:
        print("ERROR: Could not import CleanUNet2. Check models/cleanunet2/ directory")
        sys.exit(1)

class AudioDenoiser:
    def __init__(self, model_path, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        print(f"Using device: {self.device}")
        
        # Audio settings
        self.sample_rate = 16000
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Spectrogram transform (match training config)
        self.spec_transform = T.Spectrogram(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=1.0,  # Match training power setting
            normalized=True,
            center=False,
            return_complex=False
        ).to(self.device)
        
        # Warm up the model
        self.warmup_model()
        
    def load_model(self, model_path):
        """Load the trained CleanUNet2 model"""
        print(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            sys.exit(1)
        
        # Initialize model with exact training configuration
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
            cleanspecnet_input_channels=513,
            cleanspecnet_num_conv_layers=3,
            cleanspecnet_kernel_size=3,
            cleanspecnet_stride=1,
            cleanspecnet_num_attention_layers=2,
            cleanspecnet_num_heads=4,
            cleanspecnet_hidden_dim=128,
            cleanspecnet_dropout=0.1
        ).to(self.device)
        
        # Load state dict with proper key handling
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if "state_dict" in checkpoint:
            # Handle framework checkpoint format
            state_dict = checkpoint["state_dict"]
            # Add 'model.' prefix to match wrapper structure
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = f"model.{key}"
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
            
            print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"Training loss: {checkpoint.get('loss', 'unknown')}")
            
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def warmup_model(self):
        """Warm up the model with dummy data"""
        print("Warming up model...")
        dummy_waveform = torch.randn(1, 1, self.sample_rate).to(self.device)
        dummy_spec = self.spec_transform(dummy_waveform.squeeze(1))
        dummy_spec = dummy_spec.unsqueeze(0)
        
        with torch.no_grad():
            _ = self.model(dummy_waveform, dummy_spec)
        print("Warm-up complete")
    
    def denoise_audio(self, waveform):
        """Denoise a single audio waveform"""
        # Ensure proper shape [1, 1, T]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(self.device)
        
        # Compute spectrogram
        spectrogram = self.spec_transform(waveform.squeeze(1))  # [1, F, T]
        
        # Denoise
        with torch.no_grad():
            output = self.model(waveform, spectrogram)
            
            # Handle tuple output
            if isinstance(output, tuple):
                denoised = output[0]  # Enhanced audio
            else:
                denoised = output
        
        # Return to CPU and clip
        return denoised.squeeze().cpu().clamp(-1, 1)
    
    def process_file(self, input_path, output_path=None, play_audio=False):
        """Process a single audio file"""
        print(f"\nProcessing: {os.path.basename(input_path)}")
        
        if not os.path.exists(input_path):
            print(f"ERROR: Input file not found: {input_path}")
            return None
        
        # Load audio
        waveform, sr = torchaudio.load(input_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        duration = waveform.shape[-1] / self.sample_rate
        
        # Denoise
        print("Denoising...")
        start_time = time.time()
        denoised = self.denoise_audio(waveform)
        
        # Synchronize if using CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        processing_time = time.time() - start_time
        rtf = processing_time / duration
        
        print(f"Inference time: {processing_time:.4f} sec")
        print(f"Audio duration: {duration:.2f} sec")
        print(f"Real-Time Factor (RTF): {rtf:.4f} -> {'Real-time' if rtf < 1 else 'Too slow'}")
        
        # Save if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, denoised.unsqueeze(0), self.sample_rate)
            print(f"Saved denoised audio to: {output_path}")
        
        # Play audio for comparison
        if play_audio and AUDIO_PLAYBACK_AVAILABLE:
            self.play_comparison(waveform.squeeze().numpy(), denoised.numpy())
        
        return denoised
    
    def play_comparison(self, original, denoised):
        """Play original and denoised audio for comparison"""
        print("\nAudio Comparison:")
        print("1. Playing ORIGINAL (noisy)...")
        sd.play(original, self.sample_rate)
        sd.wait()
        
        time.sleep(0.5)
        
        print("2. Playing DENOISED...")
        sd.play(denoised, self.sample_rate)
        sd.wait()
        
        print("Comparison complete!")
    
    def test_from_directory(self, audio_dir, num_samples=15, save_results=True):
        """Test on audio files from a directory"""
        print(f"\nTesting {num_samples} samples from: {audio_dir}")
        
        if not os.path.exists(audio_dir):
            print(f"ERROR: Directory not found: {audio_dir}")
            return
        
        # Find all wav files
        wav_files = []
        for file in os.listdir(audio_dir):
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(audio_dir, file))
        
        if not wav_files:
            print(f"ERROR: No .wav files found in {audio_dir}")
            return
        
        print(f"Found {len(wav_files)} audio files")
        
        # Random sample
        num_samples = min(num_samples, len(wav_files))
        selected_files = random.sample(wav_files, num_samples)
        
        # Results directory
        results_dir = "test_results_directory"
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
        
        total_rtf = []
        success_count = 0
        
        for i, audio_file in enumerate(selected_files):
            print(f"\n" + "="*60)
            print(f"Sample {i+1}/{num_samples}")
            print(f"File: {os.path.basename(audio_file)}")
            print("="*60)
            
            # Output path
            output_path = None
            if save_results:
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                output_path = os.path.join(results_dir, f"denoised_{base_name}.wav")
                # Also save original for comparison
                original_path = os.path.join(results_dir, f"original_{base_name}.wav")
                
            try:
                # Load original for saving
                if save_results:
                    orig_waveform, orig_sr = torchaudio.load(audio_file)
                    if orig_sr != self.sample_rate:
                        orig_waveform = torchaudio.functional.resample(orig_waveform, orig_sr, self.sample_rate)
                    if orig_waveform.shape[0] > 1:
                        orig_waveform = orig_waveform.mean(dim=0, keepdim=True)
                    torchaudio.save(original_path, orig_waveform, self.sample_rate)
                
                # Process
                start_time = time.time()
                denoised = self.process_file(audio_file, output_path, play_audio=False)
                
                if denoised is not None:
                    # Track RTF
                    duration = torchaudio.load(audio_file)[0].shape[-1] / self.sample_rate
                    processing_time = time.time() - start_time
                    rtf = processing_time / duration
                    total_rtf.append(rtf)
                    success_count += 1
                    
                    print(f"SUCCESS: Processed {os.path.basename(audio_file)}")
                else:
                    print(f"FAILED: Could not process {os.path.basename(audio_file)}")
                
            except Exception as e:
                print(f"ERROR processing {os.path.basename(audio_file)}: {e}")
                continue
        
        # Summary
        if total_rtf:
            avg_rtf = np.mean(total_rtf)
            print(f"\n" + "="*60)
            print(f"SUMMARY")
            print("="*60)
            print(f"Samples processed: {success_count}/{num_samples}")
            print(f"Average RTF: {avg_rtf:.4f}")
            print(f"Real-time capable: {'Yes' if avg_rtf < 1 else 'No'}")
            if save_results:
                print(f"Results saved to: {results_dir}/")
                print(f"Download with: scp -r user@server:path/{results_dir}/ ./")

def main():
    # Configuration - UPDATE THESE PATHS
    model_path = "outputs/cleanunet2/cleanunet2_best.pth"  # Your trained model
    test_audio_dir = "data/testdata/noisy"  # Your directory with test audio files
    num_samples = 15  # Number of samples to test
    
    print("Audio Denoising Test Suite")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Test audio directory: {test_audio_dir}")
    print(f"Number of samples: {num_samples}")
    print("=" * 60)
    
    try:
        # Initialize denoiser
        denoiser = AudioDenoiser(model_path)
        
        # Test on audio files from directory
        denoiser.test_from_directory(test_audio_dir, num_samples, save_results=True)
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        print("Make sure you have:")
        print(f"1. Trained model at: {model_path}")
        print(f"2. Test audio files in: {test_audio_dir}")
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
