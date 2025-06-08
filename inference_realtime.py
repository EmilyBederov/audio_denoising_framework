# inference_realtime_cpu.py - CPU-only inference optimized for hearing aids

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa
import torchaudio
import torchaudio.transforms as T
import time
import os
import sys
from pathlib import Path
from collections import deque
import threading
import queue

# Add project paths for imports
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from core.model_factory import ModelFactory

class HearingAidProcessor:
    def __init__(self, model_path, config_path, frame_size=512, hop_length=256):
        """
        Initialize CPU-only real-time inference optimized for hearing aids
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model config file
            frame_size: Frame size for processing (samples) - keep small for low latency
            hop_length: Hop length for overlapping frames
        """
        
        # FORCE CPU usage - critical for hearing aid deployment
        torch.set_num_threads(1)  # Single thread for deterministic latency
        torch.set_grad_enabled(False)  # Disable gradients globally
        self.device = torch.device('cpu')
        
        print(f"🎧 Hearing Aid Processor initialized on CPU")
        print(f"   Frame size: {frame_size} samples ({frame_size/16000*1000:.1f}ms)")
        print(f"   Hop length: {hop_length} samples ({hop_length/16000*1000:.1f}ms)")
        
        self.sample_rate = 16000
        self.frame_size = frame_size
        self.hop_length = hop_length
        
        # Load config
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load and optimize model
        self.model = self.load_and_optimize_model(model_path)
        
        # Audio processing components
        self.setup_audio_processing()
        
        # Buffers for real-time processing
        self.input_buffer = deque(maxlen=frame_size * 2)  # Larger buffer for stability
        self.output_buffer = deque()
        
        # Performance monitoring
        self.latency_history = deque(maxlen=100)
        self.frame_count = 0
        
        # Warm up the model
        self.warmup_model()
        
    def load_and_optimize_model(self, model_path):
        """Load model and optimize for CPU inference"""
        print(f"📦 Loading model from: {model_path}")
        
        # Create model using ModelFactory
        model = ModelFactory.create_model("cleanunet2", self.config)
        model = model.to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print(f"   Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # CPU optimizations
        print("⚡ Applying CPU optimizations...")
        
        # 1. Script the model for faster execution
        dummy_audio = torch.randn(1, 1, self.frame_size)
        dummy_spec = torch.randn(1, 513, self.frame_size // 256 + 1)
        
        try:
            model = torch.jit.trace(model, (dummy_audio, dummy_spec))
            print("    Model traced successfully")
        except Exception as e:
            print(f"    Model tracing failed: {e}, using eager mode")
        
        # 2. Optimize for inference
        model = torch.jit.optimize_for_inference(model) if hasattr(torch.jit, 'optimize_for_inference') else model
        
        return model
    
    def setup_audio_processing(self):
        """Setup audio processing transforms"""
        self.spec_transform = T.Spectrogram(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=1.0,
            normalized=True,
            center=False,
            return_complex=False
        )
        
    def warmup_model(self):
        """Warm up the model to avoid first-inference latency"""
        print(" Warming up model...")
        
        for _ in range(5):  # Multiple warmup runs
            dummy_audio = torch.randn(1, 1, self.frame_size)
            dummy_spec = self.spec_transform(dummy_audio.squeeze(1))
            
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(dummy_audio, dummy_spec)
            warmup_time = (time.time() - start_time) * 1000
            
        print(f"   ✅ Warmup completed, last inference: {warmup_time:.2f}ms")
    
    def process_frame(self, audio_frame):
        """
        Process a single frame of audio with latency monitoring
        
        Args:
            audio_frame: numpy array of audio samples [frame_size]
            
        Returns:
            enhanced_frame: numpy array of enhanced audio samples
        """
        
        start_time = time.time()
        
        # Convert to tensor and add batch/channel dimensions
        audio_tensor = torch.FloatTensor(audio_frame).unsqueeze(0).unsqueeze(0)  # [1, 1, frame_size]
        
        # Compute spectrogram
        spec_tensor = self.spec_transform(audio_tensor.squeeze(1))  # [1, freq, time]
        
        # Inference with no gradient computation
        with torch.no_grad():
            enhanced_audio, _ = self.model(audio_tensor, spec_tensor)
            
        # Convert back to numpy
        enhanced_frame = enhanced_audio.squeeze().numpy()
        
        # Measure and track latency
        inference_time = time.time() - start_time
        latency_ms = inference_time * 1000
        self.latency_history.append(latency_ms)
        self.frame_count += 1
        
        # Warning for high latency (critical for hearing aids)
        if latency_ms > 20:  # 20ms is pushing it for hearing aids
            print(f"⚠️ High latency: {latency_ms:.2f}ms (frame {self.frame_count})")
        
        return enhanced_frame
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.latency_history:
            return {}
        
        latencies = list(self.latency_history)
        return {
            'avg_latency_ms': np.mean(latencies),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies),
            'std_latency_ms': np.std(latencies),
            'frames_processed': self.frame_count,
            'real_time_factor': np.mean(latencies) / (self.hop_length / self.sample_rate * 1000)
        }
    
    def process_file_streaming(self, input_file, output_file, chunk_size=None):
        """
        Process audio file in streaming fashion (simulating real-time)
        
        Args:
            input_file: Path to input audio file
            output_file: Path to save enhanced audio
            chunk_size: Size of processing chunks (None = use hop_length)
        """
        
        if chunk_size is None:
            chunk_size = self.hop_length
            
        print(f"🎵 Processing file: {input_file}")
        print(f"   Chunk size: {chunk_size} samples ({chunk_size/self.sample_rate*1000:.1f}ms)")
        
        # Load audio
        audio, sr = librosa.load(input_file, sr=self.sample_rate)
        print(f"   Duration: {len(audio)/self.sample_rate:.2f} seconds")
        
        # Process in real-time chunks
        enhanced_audio = []
        
        for i in range(0, len(audio), chunk_size):
            # Get chunk
            chunk = audio[i:i + chunk_size]
            
            # Pad if necessary
            if len(chunk) < self.frame_size:
                padded_chunk = np.pad(chunk, (0, self.frame_size - len(chunk)))
            else:
                # Use sliding window for larger chunks
                padded_chunk = chunk[:self.frame_size]
            
            # Process
            enhanced_chunk = self.process_frame(padded_chunk)
            
            # Take only the new samples
            if len(chunk) >= chunk_size:
                enhanced_audio.extend(enhanced_chunk[:chunk_size])
            else:
                enhanced_audio.extend(enhanced_chunk[:len(chunk)])
        
        # Save enhanced audio
        enhanced_audio = np.array(enhanced_audio[:len(audio)])  # Trim to original length
        sf.write(output_file, enhanced_audio, self.sample_rate)
        
        # Performance summary
        stats = self.get_performance_stats()
        print(f"\n📊 Performance Summary:")
        print(f"   Average latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"   Max latency: {stats['max_latency_ms']:.2f}ms")
        print(f"   Real-time factor: {stats['real_time_factor']:.3f}")
        print(f"   Real-time capable: {' Yes' if stats['avg_latency_ms'] < 20 else ' No'}")
        
        return enhanced_audio
    
    def process_real_time_stream(self, input_queue, output_queue, stop_event):
        """
        Process real-time audio stream (for actual hearing aid deployment)
        
        Args:
            input_queue: Queue containing input audio chunks
            output_queue: Queue for output enhanced audio chunks  
            stop_event: Threading event to stop processing
        """
        
        print(" Starting real-time stream processing...")
        
        while not stop_event.is_set():
            try:
                # Get audio chunk from input queue
                audio_chunk = input_queue.get(timeout=0.001)  # 1ms timeout
                
                if audio_chunk is None:  # Termination signal
                    break
                
                # Add samples to buffer
                self.input_buffer.extend(audio_chunk)
                
                # Process if we have enough samples
                while len(self.input_buffer) >= self.frame_size:
                    # Extract frame
                    frame = np.array(list(self.input_buffer)[:self.frame_size])
                    
                    # Process frame
                    enhanced_frame = self.process_frame(frame)
                    
                    # Add to output queue (only new samples)
                    output_queue.put(enhanced_frame[:self.hop_length])
                    
                    # Remove processed samples from buffer
                    for _ in range(self.hop_length):
                        if self.input_buffer:
                            self.input_buffer.popleft()
                            
            except queue.Empty:
                continue
            except Exception as e:
                print(f" Error in real-time processing: {e}")
                break
                
        print("⏹ Real-time stream processing stopped")

# Example usage for hearing aid application
def main():
    # Configuration
    model_path = "outputs/cleanunet2/cleanunet2_best.pth"
    config_path = "configs/configs-cleanunet2/cleanunet2-config.yaml"
    
    # Initialize processor with hearing aid optimized settings
    processor = HearingAidProcessor(
        model_path=model_path,
        config_path=config_path,
        frame_size=512,    # ~32ms at 16kHz - good balance for hearing aids
        hop_length=256     # ~16ms hop - ensures low latency
    )
    
    # Test with audio file
    input_file = "test_input.wav"
    output_file = "test_output_enhanced.wav"
    
    if os.path.exists(input_file):
        enhanced = processor.process_file_streaming(input_file, output_file)
        print(f" Enhanced audio saved to: {output_file}")
    else:
        print(f" Input file not found: {input_file}")

if __name__ == "__main__":
    main()
