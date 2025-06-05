#!/usr/bin/env python3
"""
Real-time audio denoising with CleanUNet2
Modified to process matching noisy/clean pairs and create tar archive
"""
import time
import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import random
import sys
from pathlib import Path
import argparse
import yaml
import tarfile
import shutil

# Try to import sounddevice, but make it optional
try:
    import sounddevice as sd
    AUDIO_PLAYBACK_AVAILABLE = True
except (ImportError, OSError) as e:
    print("WARNING: Audio playback not available")
    print(f"Reason: {e}")
    print("Will save audio files instead for manual listening")
    AUDIO_PLAYBACK_AVAILABLE = False

# Add project paths for proper imports
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import from the framework
from core.model_factory import ModelFactory

class AudioDenoiser:
    def __init__(self, model_path, config_path, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        print(f"Using device: {self.device}")
        
        # Audio settings (match training config)
        self.sample_rate = 16000
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Spectrogram transform (match training config)
        self.spec_transform = T.Spectrogram(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=1.0,
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
        
        # Create model using ModelFactory
        model = ModelFactory.create_model("cleanunet2", self.config)
        model = model.to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            print(f"Loaded checkpoint from epoch: {epoch}, loss: {loss}")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Load state dict (handle wrapper prefix if needed)
        try:
            model.load_state_dict(state_dict, strict=True)
            print(" Model loaded with strict=True")
        except RuntimeError as e:
            print(f" Strict loading failed, trying with model. prefix...")
            # Try adding 'model.' prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('model.'):
                    new_key = f"model.{key}"
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            try:
                model.load_state_dict(new_state_dict, strict=False)
                print(" Model loaded with model. prefix")
            except RuntimeError:
                print(" Failed to load model. Check checkpoint compatibility.")
                raise
        
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
    
    def load_and_preprocess_audio(self, audio_path):
        """Load and preprocess audio file"""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform
    
    def process_matching_datasets(self, noisy_dir, clean_dir, denoise_dir, num_samples=15):
        """Process matching noisy/clean pairs and generate denoised versions"""
        print(f"\nProcessing matching datasets:")
        print(f"Noisy dir: {noisy_dir}")
        print(f"Clean dir: {clean_dir}")
        print(f"Denoise dir: {denoise_dir}")
        print(f"Number of samples: {num_samples}")
        print("="*80)
        
        # Ensure directories exist
        os.makedirs(denoise_dir, exist_ok=True)
        
        # Find matching files
        noisy_files = set(f for f in os.listdir(noisy_dir) if f.lower().endswith('.wav'))
        clean_files = set(f for f in os.listdir(clean_dir) if f.lower().endswith('.wav'))
        
        # Find common files
        common_files = list(noisy_files.intersection(clean_files))
        
        if not common_files:
            print("ERROR: No matching files found between noisy and clean directories")
            return [], []
        
        print(f"Found {len(common_files)} matching files")
        
        # Select random subset
        num_samples = min(num_samples, len(common_files))
        selected_files = random.sample(common_files, num_samples)
        
        print(f"\nSelected {num_samples} files for processing:")
        for i, filename in enumerate(selected_files, 1):
            print(f"  {i:2d}. {filename}")
        
        # Process files
        total_rtf = []
        processed_files = []
        success_count = 0
        
        for i, filename in enumerate(selected_files, 1):
            print(f"\n" + "="*80)
            print(f"Processing {i}/{num_samples}: {filename}")
            print("="*80)
            
            # File paths
            noisy_path = os.path.join(noisy_dir, filename)
            clean_path = os.path.join(clean_dir, filename)
            denoise_path = os.path.join(denoise_dir, filename)
            
            try:
                # Verify clean file exists (we know noisy exists from matching)
                if not os.path.exists(clean_path):
                    print(f"WARNING: Clean file not found: {clean_path}")
                    continue
                
                # Load noisy audio
                print("Loading noisy audio...")
                noisy_waveform = self.load_and_preprocess_audio(noisy_path)
                duration = noisy_waveform.shape[-1] / self.sample_rate
                print(f"Audio duration: {duration:.2f} seconds")
                
                # Denoise
                print("Denoising...")
                start_time = time.time()
                denoised_waveform = self.denoise_audio(noisy_waveform)
                
                # Synchronize if using CUDA
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                processing_time = time.time() - start_time
                rtf = processing_time / duration
                
                print(f"Inference time: {processing_time:.4f} sec")
                print(f"Real-Time Factor (RTF): {rtf:.4f} -> {' Real-time' if rtf < 1 else ' Too slow'}")
                
                # Save denoised audio
                torchaudio.save(denoise_path, denoised_waveform.unsqueeze(0), self.sample_rate)
                print(f" Saved denoised audio: {denoise_path}")
                
                # Track successful processing
                total_rtf.append(rtf)
                processed_files.append(filename)
                success_count += 1
                
            except Exception as e:
                print(f" ERROR processing {filename}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary
        print(f"\n" + "="*80)
        print("PROCESSING SUMMARY")
        print("="*80)
        print(f"Files processed successfully: {success_count}/{num_samples}")
        
        if total_rtf:
            avg_rtf = np.mean(total_rtf)
            min_rtf = np.min(total_rtf)
            max_rtf = np.max(total_rtf)
            
            print(f"\nPerformance Metrics:")
            print(f"  Average RTF: {avg_rtf:.4f}")
            print(f"  Min RTF: {min_rtf:.4f}")
            print(f"  Max RTF: {max_rtf:.4f}")
            print(f"  Real-time capable: {' Yes' if avg_rtf < 1 else ' No'}")
            
            # Performance rating
            if avg_rtf < 0.5:
                print(f"  Performance: Excellent (2x faster than real-time)")
            elif avg_rtf < 1.0:
                print(f"  Performance: Good (real-time capable)")
            else:
                print(f"  Performance: Needs optimization")
        
        return processed_files, total_rtf
    
    def create_results_archive(self, noisy_dir, clean_dir, denoise_dir, processed_files, output_archive="denoising_results.tar"):
        """Create a tar archive with the processed files from all three directories"""
        print(f"\n" + "="*80)
        print("CREATING RESULTS ARCHIVE")
        print("="*80)
        
        # Create temporary directory structure for the archive
        temp_dir = "temp_archive"
        temp_noisy = os.path.join(temp_dir, "noisy")
        temp_clean = os.path.join(temp_dir, "clean")
        temp_denoise = os.path.join(temp_dir, "denoise")
        
        # Clean up any existing temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Create temporary directories
        os.makedirs(temp_noisy, exist_ok=True)
        os.makedirs(temp_clean, exist_ok=True)
        os.makedirs(temp_denoise, exist_ok=True)
        
        print(f"Copying {len(processed_files)} processed files...")
        
        # Copy processed files to temporary structure
        for filename in processed_files:
            # Source paths
            src_noisy = os.path.join(noisy_dir, filename)
            src_clean = os.path.join(clean_dir, filename)
            src_denoise = os.path.join(denoise_dir, filename)
            
            # Destination paths
            dst_noisy = os.path.join(temp_noisy, filename)
            dst_clean = os.path.join(temp_clean, filename)
            dst_denoise = os.path.join(temp_denoise, filename)
            
            # Copy files
            try:
                if os.path.exists(src_noisy):
                    shutil.copy2(src_noisy, dst_noisy)
                if os.path.exists(src_clean):
                    shutil.copy2(src_clean, dst_clean)
                if os.path.exists(src_denoise):
                    shutil.copy2(src_denoise, dst_denoise)
                print(f"  {filename}")
            except Exception as e:
                print(f"   {filename}: {e}")
        
        # Create tar archive
        print(f"\nCreating archive: {output_archive}")
        with tarfile.open(output_archive, "w") as tar:
            tar.add(temp_dir, arcname="denoising_results")
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        # Get archive size
        archive_size = os.path.getsize(output_archive) / (1024 * 1024)  # MB
        
        print(f" Archive created successfully!")
        print(f" Archive: {output_archive}")
        print(f" Size: {archive_size:.1f} MB")
        print(f" Contains: {len(processed_files)} files × 3 versions = {len(processed_files) * 3} total files")
        
        return output_archive

def main():
    parser = argparse.ArgumentParser(description='CleanUNet2 Real-time Audio Denoising with Archive Creation')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, 
                        default='configs/configs-cleanunet2/cleanunet2-config.yaml',
                        help='Path to model config file')
    parser.add_argument('--noisy_dir', type=str, default='data/testdata/noisy',
                        help='Directory with noisy audio files')
    parser.add_argument('--clean_dir', type=str, default='data/testdata/clean',
                        help='Directory with clean audio files')
    parser.add_argument('--denoise_dir', type=str, default='data/testdata/denoise',
                        help='Directory to save denoised audio files')
    parser.add_argument('--num_samples', type=int, default=15,
                        help='Number of samples to process')
    parser.add_argument('--output_archive', type=str, default='denoising_results.tar',
                        help='Output tar archive filename')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    print("CleanUNet2 Real-time Audio Denoising")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Noisy directory: {args.noisy_dir}")
    print(f"Clean directory: {args.clean_dir}")
    print(f"Denoise directory: {args.denoise_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output archive: {args.output_archive}")
    print("=" * 80)
    
    # Validate input directories
    if not os.path.exists(args.noisy_dir):
        print(f"ERROR: Noisy directory not found: {args.noisy_dir}")
        return
    
    if not os.path.exists(args.clean_dir):
        print(f"ERROR: Clean directory not found: {args.clean_dir}")
        return
    
    try:
        # Initialize denoiser
        denoiser = AudioDenoiser(args.model, args.config, device=args.device)
        
        # Process matching datasets
        processed_files, rtf_list = denoiser.process_matching_datasets(
            args.noisy_dir,
            args.clean_dir, 
            args.denoise_dir,
            args.num_samples
        )
        
        if processed_files:
            # Create results archive
            archive_path = denoiser.create_results_archive(
                args.noisy_dir,
                args.clean_dir,
                args.denoise_dir,
                processed_files,
                args.output_archive
            )
            
            print(f"\n Successfully processed {len(processed_files)} files!")
            print(f" Results archived in: {archive_path}")
            print(f"\nTo extract: tar -xf {archive_path}")
        else:
            print("\n No files were processed successfully")
            
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        print("Make sure you have:")
        print(f"1. Trained model at: {args.model}")
        print(f"2. Config file at: {args.config}")
        print(f"3. Data directories with matching files")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
