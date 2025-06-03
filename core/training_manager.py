# core/training_manager.py6xy66x
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import torchaudio
from tqdm import tqdm
from pathlib import Path
import time

from core.model_factory import ModelFactory
from core.metrics import calculate_pesq, calculate_stoi, calculate_snr

class TrainingManager:
    """Manager for training, evaluation, and inference"""
    
    def __init__(self, model_name: str, config: Dict[str, Any], device: str = 'cuda'):
        """
        Initialize TrainingManager.
        
        Args:
            model_name: Name of the model to use
            config: Configuration dictionary
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        # ADD: Print device info for debugging
        print(f"Using device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
        
        # Create model
        self.model = ModelFactory.create_model(model_name, config)
        self.model.to(self.device)  # This is correct
        
        # ADD: Verify model is on GPU
        print(f"Model device: {next(self.model.parameters()).device}")
        
        # Configure metrics - FIXED with proper error handling
        self.metrics = {
            'pesq': calculate_pesq,
            'stoi': calculate_stoi,
            'snr': calculate_snr
        }
        
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, 
              epochs: int = None, checkpoint_dir: str = None):
        """
        Train the model.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional)
            epochs: Number of epochs to train (overrides config)
            checkpoint_dir: Directory to save checkpoints
        """
        if epochs is None:
            epochs = self.config.get('epochs', 100)
            
        if checkpoint_dir is None:
            checkpoint_dir = self.config.get('save_path', 'outputs')
            
        # Create optimizer
        learning_rate = self.config.get('learning_rate', 0.0001)
        weight_decay = self.config.get('weight_decay', 0)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
        # Get loss function
        loss_module = self._get_loss_module()
        
        # Create checkpoint directory
        checkpoint_dir = Path(checkpoint_dir) / self.model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # Training
            self.model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}")
            for batch_idx, batch in enumerate(progress_bar):
                # Get inputs and targets - FIX: Move to GPU here
                inputs, targets = self._process_batch(batch)
                
                # ADD: Verify data is on GPU
                if batch_idx == 0:  # Only print for first batch
                    print(f"Input device: {inputs[0].device if inputs else 'None'}")
                    print(f"Target device: {targets.device}")
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(*inputs)
                
                # Handle model output - CleanUNet2 might return tuple (audio, spec)
                if isinstance(outputs, tuple):
                    # If model returns (enhanced_audio, enhanced_spec), use audio for loss
                    enhanced_audio = outputs[0]
                else:
                    # If model returns only enhanced audio
                    enhanced_audio = outputs
                
                # Calculate loss
                loss = loss_module(enhanced_audio, targets)
                
                # ADD DEBUG for suspicious loss values
                if batch_idx == 0 and epoch == 1:  # Only first batch of first epoch
                    print(f"\n=== REAL TRAINING DATA DEBUG ===")
                    print(f"  Target (clean) audio: min={targets.min():.4f}, max={targets.max():.4f}, mean={targets.mean():.4f}")
                    print(f"  Enhanced audio: min={enhanced_audio.min():.4f}, max={enhanced_audio.max():.4f}, mean={enhanced_audio.mean():.4f}")
                    print(f"  Target std: {targets.std():.4f}")
                    print(f"  Enhanced std: {enhanced_audio.std():.4f}")
                    print(f"  Raw L1 difference: {torch.abs(enhanced_audio - targets).mean():.6f}")
                    print(f"  Loss value: {loss.item():.6f}")
                    print(f"  Target range: [{targets.min():.6f}, {targets.max():.6f}]")
                    print(f"  Enhanced range: [{enhanced_audio.min():.6f}, {enhanced_audio.max():.6f}]")
                    print(f"  Audio shapes: targets={targets.shape}, enhanced={enhanced_audio.shape}")
                    print(f"  Are outputs close to targets? {torch.allclose(enhanced_audio, targets, atol=0.01)}")
                    print("=== END DEBUG ===\n")
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update metrics
                batch_loss = loss.item()
                train_loss += batch_loss
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f"{batch_loss:.4f}"})
                
            # Calculate epoch metrics
            train_loss /= len(train_dataloader)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            
            # Validation
            if val_dataloader is not None:
                val_loss, val_metrics = self.evaluate(val_dataloader, loss_module)
                print(f"Validation: loss={val_loss:.4f}, PESQ={val_metrics.get('pesq', 0):.4f}, "
                      f"STOI={val_metrics.get('stoi', 0):.4f}, SNR={val_metrics.get('snr', 0):.4f}")
                
                # Save if best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = checkpoint_dir / f"{self.model_name}_best.pth"
                    self.model.save_checkpoint(
                        checkpoint_path, 
                        optimizer=optimizer, 
                        epoch=epoch, 
                        loss=val_loss
                    )
                    print(f"Saved best model with validation loss: {val_loss:.4f}")
                
            # Save regular checkpoint
            checkpoint_path = checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pth"
            self.model.save_checkpoint(
                checkpoint_path, 
                optimizer=optimizer, 
                epoch=epoch, 
                loss=train_loss
            )
                
        # Save final model
        checkpoint_path = checkpoint_dir / f"{self.model_name}_final.pth"
        self.model.save_checkpoint(
            checkpoint_path, 
            optimizer=optimizer, 
            epoch=epochs, 
            loss=train_loss
        )
        print(f"Training completed. Final model saved to {checkpoint_path}")
    
    def evaluate(self, dataloader: DataLoader, loss_module=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            loss_module: Loss function (if None, uses default)
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        if loss_module is None:
            loss_module = self._get_loss_module()
            
        self.model.eval()
        total_loss = 0.0
        all_metrics = {name: [] for name in self.metrics.keys()}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Get inputs and targets - FIX: Move to GPU here
                inputs, targets = self._process_batch(batch)
                
                # Forward pass
                outputs = self.model(*inputs)
                
                # Handle model output - CleanUNet2 might return tuple (audio, spec)
                if isinstance(outputs, tuple):
                    # If model returns (enhanced_audio, enhanced_spec), use audio for loss
                    enhanced_audio = outputs[0]
                else:
                    # If model returns only enhanced audio
                    enhanced_audio = outputs
                
                # Calculate loss
                loss = loss_module(enhanced_audio, targets)
                total_loss += loss.item()
                
                # Calculate metrics
                sample_rate = self.config.get('sample_rate', 16000)
                for metric_name, metric_func in self.metrics.items():
                    if metric_name in ['pesq', 'stoi']:
                        metric_value = metric_func(targets, enhanced_audio, sample_rate)
                    else:
                        metric_value = metric_func(targets, enhanced_audio)
                    all_metrics[metric_name].append(metric_value)
        
        # Average metrics
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {name: sum(values) / len(values) for name, values in all_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def inference(self, input_path: str, output_path: Optional[str] = None):
        """
        Run inference on a single audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save output (if None, generates automatically)
            
        Returns:
            Path to output file
        """
        import torchaudio.transforms as T
        
        # Load audio
        waveform, sample_rate = torchaudio.load(input_path)
        
        # Resample if necessary
        target_sr = self.config.get('sample_rate', 16000)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Add batch dimension and move to device
        waveform = waveform.unsqueeze(0).to(self.device)  # [1, 1, T]
        
        # Compute spectrogram (same as training)
        spec_transform = T.Spectrogram(
            n_fft=self.config.get('n_fft', 1024),
            hop_length=self.config.get('hop_length', 256),
            win_length=self.config.get('win_length', 1024),
            power=self.config.get('power', 1.0),
            normalized=True,
            center=False
        ).to(self.device)
        
        spectrogram = spec_transform(waveform.squeeze(0))  # [F, T]
        spectrogram = spectrogram.unsqueeze(0)  # [1, F, T]
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            enhanced = self.model(waveform, spectrogram)
            
            # Handle tuple output (enhanced_audio, enhanced_spec)
            if isinstance(enhanced, tuple):
                enhanced = enhanced[0]  # Get the audio output
        
        # Generate output path if not provided
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = input_path_obj.parent / f"{input_path_obj.stem}_enhanced{input_path_obj.suffix}"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save enhanced audio
        enhanced = enhanced.cpu().squeeze()
        torchaudio.save(output_path, enhanced.unsqueeze(0), sample_rate)
        
        return str(output_path)
    
    def _process_batch(self, batch):
        """Process a batch of data based on model requirements"""
        # Handle both tuple and list formats
        if (isinstance(batch, (tuple, list)) and len(batch) == 2):
            # Simple case: (noisy, clean) or [noisy, clean]
            noisy, clean = batch
            return [noisy.to(self.device)], clean.to(self.device)
        elif (isinstance(batch, (tuple, list)) and len(batch) == 4):
            # CleanUNet2 case: (clean_audio, clean_spec, noisy_audio, noisy_spec)
            clean_audio, clean_spec, noisy_audio, noisy_spec = batch
            inputs = [noisy_audio.to(self.device), noisy_spec.to(self.device)]
            return inputs, clean_audio.to(self.device)
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'unknown'}")
    
    def _get_loss_module(self):
        """Get the loss function based on configuration"""
        loss_type = self.config.get('loss_type', 'l1')
        
        if loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
