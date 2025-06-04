import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from tqdm import tqdm

# CleanUNet2 loss imports
try:
    from models.cleanunet2.stft_loss import MultiResolutionSTFTLoss, CleanUNet2Loss
    CLEANUNET2_AVAILABLE = True
except ImportError:
    CLEANUNET2_AVAILABLE = False
    print("Warning: CleanUNet2 STFT loss not available, falling back to L1 loss")

class TrainingManager:
    def __init__(self, model_name, config, device='cuda'):
        self.model_name = model_name
        self.config = config
        self.device = torch.device(device)
        
        # Create model
        from core.model_factory import ModelFactory
        self.model = ModelFactory.create_model(model_name, config)
        self.model = self.model.to(self.device)
        
        # Initialize proper loss function based on model
        self._setup_loss_function()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup logger
        self.logger = logging.getLogger("TrainingManager")
        
        print(f"Training manager created for {model_name} on {self.device}")
    
    def _setup_loss_function(self):
        """Setup the appropriate loss function based on model type"""
        if self.model_name == 'cleanunet2' and CLEANUNET2_AVAILABLE:
            # Use CleanUNet2 paper loss (L1 + Multi-resolution STFT)
            mrstft_loss = MultiResolutionSTFTLoss(
                fft_sizes=[512, 1024, 2048],
                hop_sizes=[50, 120, 240],
                win_lengths=[240, 600, 1200],
                window="hann_window",
                sc_lambda=0.5,
                mag_lambda=0.5,
                band="full"
            ).to(self.device)
            
            self.loss_fn = CleanUNet2Loss(
                ell_p=1,
                ell_p_lambda=1.0,
                stft_lambda=1.0,
                mrstftloss=mrstft_loss
            )
            print("✅ Using CleanUNet2 paper loss (L1 + Multi-resolution STFT)")
        else:
            self.loss_fn = torch.nn.L1Loss()
            print("✅ Using default L1 loss")
    
    def _setup_optimizer(self):
        """Setup optimizer based on config"""
        lr = self.config.get('learning_rate', 0.0001)
        weight_decay = self.config.get('weight_decay', 0.0)
        
        if self.model_name == 'cleanunet2':
            # Use CleanUNet2 paper optimizer settings
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        return optimizer
    
    def compute_loss(self, outputs, targets):
        """Compute loss with proper handling for different model outputs"""
        if self.model_name == 'cleanunet2' and CLEANUNET2_AVAILABLE:
            # CleanUNet2 returns tuple (enhanced_audio, enhanced_spec)
            if isinstance(outputs, tuple):
                enhanced_audio, enhanced_spec = outputs
            else:
                enhanced_audio = outputs
            
            # CleanUNet2Loss expects (clean_audio, denoised_audio)
            loss = self.loss_fn(targets, enhanced_audio)
            return loss
        else:
            # Default behavior for other models
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take first output if tuple
            loss = self.loss_fn(outputs, targets)
            return loss
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Handle different batch formats
            if len(batch) == 4:  # CleanUNet2 format: (clean_audio, clean_spec, noisy_audio, noisy_spec)
                clean_audio, clean_spec, noisy_audio, noisy_spec = batch
                inputs = (noisy_audio.to(self.device), noisy_spec.to(self.device))
                targets = clean_audio.to(self.device)
            elif len(batch) == 2:  # Standard format: (inputs, targets)
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(inputs, tuple):
                outputs = self.model(*inputs)
            else:
                outputs = self.model(inputs)
            
            # Compute loss
            loss = self.compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            current_avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{current_avg_loss:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, dataloader):
        """Run validation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation")
            
            for batch in progress_bar:
                # Handle different batch formats
                if len(batch) == 4:  # CleanUNet2 format
                    clean_audio, clean_spec, noisy_audio, noisy_spec = batch
                    inputs = (noisy_audio.to(self.device), noisy_spec.to(self.device))
                    targets = clean_audio.to(self.device)
                elif len(batch) == 2:  # Standard format
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                
                # Forward pass
                if isinstance(inputs, tuple):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                
                # Compute loss
                loss = self.compute_loss(outputs, targets)
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop"""
        epochs = self.config.get('epochs', 100)
        save_path = self.config.get('save_path', 'outputs')
        
        # Create save directory
        model_save_path = os.path.join(save_path, self.model_name)
        os.makedirs(model_save_path, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Train one epoch
            train_loss = self.train_epoch(train_dataloader)
            
            # Validate if validation data is provided
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(model_save_path, f"{self.model_name}_best.pth")
                    self.save_checkpoint(best_model_path, epoch, train_loss, val_loss)
                    print(f" New best model saved: {best_model_path}")
            else:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Save regular checkpoint every few epochs
            if epoch % self.config.get('save_every', 10) == 0:
                checkpoint_path = os.path.join(model_save_path, f"{self.model_name}_epoch_{epoch}.pth")
                self.save_checkpoint(checkpoint_path, epoch, train_loss, 
                                   val_loss if val_dataloader else None)
                print(f" Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        final_model_path = os.path.join(model_save_path, f"{self.model_name}_final.pth")
        self.save_checkpoint(final_model_path, epochs, train_loss, 
                           val_loss if val_dataloader else None)
        print(f" Training completed! Final model saved: {final_model_path}")
    
    def save_checkpoint(self, filepath, epoch, train_loss, val_loss=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        print(f"Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f" Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        return checkpoint
    
    def evaluate(self, test_dataloader):
        """Evaluate model on test data"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(test_dataloader)
        
        # Additional metrics for audio models
        metrics = {}
        
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader, desc="Evaluating")
            
            for batch in progress_bar:
                # Handle different batch formats
                if len(batch) == 4:  # CleanUNet2 format
                    clean_audio, clean_spec, noisy_audio, noisy_spec = batch
                    inputs = (noisy_audio.to(self.device), noisy_spec.to(self.device))
                    targets = clean_audio.to(self.device)
                elif len(batch) == 2:  # Standard format
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                
                # Forward pass
                if isinstance(inputs, tuple):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                
                # Compute loss
                loss = self.compute_loss(outputs, targets)
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'test_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        metrics['test_loss'] = avg_loss
        
        return avg_loss, metrics
    
    def inference(self, input_path, output_path=None):
        """Run inference on a single audio file"""
        import torchaudio
        import torchaudio.transforms as T
        
        self.model.eval()
        
        # Load audio
        waveform, sample_rate = torchaudio.load(input_path)
        target_sr = self.config.get('sample_rate', 16000)
        
        # Resample if needed
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Add batch dimension and move to device
        waveform = waveform.unsqueeze(0).to(self.device)  # [1, 1, T]
        
        # For CleanUNet2, also need spectrogram
        if self.model_name == 'cleanunet2':
            spec_transform = T.Spectrogram(
                n_fft=self.config.get('n_fft', 1024),
                hop_length=self.config.get('hop_length', 256),
                win_length=self.config.get('win_length', 1024),
                power=self.config.get('power', 1.0),
                normalized=True,
                center=False
            ).to(self.device)
            
            spectrogram = spec_transform(waveform.squeeze(1))  # [1, F, T]
            inputs = (waveform, spectrogram)
        else:
            inputs = waveform
        
        # Run inference
        with torch.no_grad():
            if isinstance(inputs, tuple):
                outputs = self.model(*inputs)
            else:
                outputs = self.model(inputs)
            
            # Extract audio output
            if isinstance(outputs, tuple):
                denoised_audio = outputs[0]  # Take audio output
            else:
                denoised_audio = outputs
        
        # Move back to CPU and remove batch dimension
        denoised_audio = denoised_audio.squeeze(0).cpu()
        
        # Save output if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, denoised_audio, target_sr)
            return output_path
        
        return denoised_audio
