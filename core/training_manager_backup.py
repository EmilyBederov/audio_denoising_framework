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
        
        # Check if model has special input handling (like UNet)
        self.model_input_type = getattr(self.model, 'model_type', 'default')
        
        # Initialize proper loss function based on model
        self._setup_loss_function()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        
        # Initialize scheduler (set by _setup_optimizer for some models)
        self.scheduler = getattr(self, 'scheduler', None)
        
        # Setup logger
        self.logger = logging.getLogger("TrainingManager")
        
        print(f"Training manager created for {model_name} on {self.device}")
        if self.model_input_type != 'default':
            print(f"Model uses {self.model_input_type} input format")
    
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
            print(" Using CleanUNet2 paper loss (L1 + Multi-resolution STFT)")
        elif self.model_name == 'unet':
            # Use EXACT LSD loss from original UNet implementation
            from models.unet.loss import LSDLoss
            self.loss_fn = LSDLoss()
            print(" Using LSD (Log-Spectral Distance) loss for UNet")
        else:
            self.loss_fn = torch.nn.L1Loss()
            print(" Using default L1 loss")
    
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
        elif self.model_name == 'unet':
            # Use EXACT optimizer settings from original UNet implementation
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                betas=(0.5, 0.9),  # EXACT from original: betas=(0.5, 0.9)
                weight_decay=weight_decay
            )
            
            # Add learning rate scheduler (EXACT from original)
            lr_decay = self.config.get('lr_decay', 0)  # Default 0 as in original
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lambda epoch: 10**(-lr_decay*epoch)
            )
            print(f"✅ Using UNet optimizer: Adam(betas=(0.5, 0.9)), lr_decay={lr_decay}")
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        return optimizer
    
    def _prepare_batch(self, batch):
        """
        Prepare batch based on model type
        
        Args:
            batch: Raw batch from dataloader
            
        Returns:
            inputs, targets prepared for the specific model
        """
        if len(batch) == 4:  # Standard format: (clean_audio, clean_spec, noisy_audio, noisy_spec)
            clean_audio, clean_spec, noisy_audio, noisy_spec = batch
            
            if self.model_name == 'cleanunet2':
                # CleanUNet2: uses (noisy_audio, noisy_spec) -> clean_audio
                inputs = (noisy_audio.to(self.device), noisy_spec.to(self.device))
                targets = clean_audio.to(self.device)
                
            elif self.model_name == 'unet':
                # UNet: uses noisy_spec -> clean_spec (spectrogram to spectrogram)
                if hasattr(self.model, 'prepare_training_batch'):
                    # Use wrapper's batch preparation
                    inputs, targets = self.model.prepare_training_batch(batch)
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    # Fallback: prepare manually
                    inputs = noisy_spec.to(self.device)
                    targets = clean_spec.to(self.device)
                    
                    # Ensure correct format [B, 1, T, F] for UNet
                    if inputs.dim() == 3:
                        inputs = inputs.unsqueeze(1).transpose(-2, -1)
                    if targets.dim() == 3:
                        targets = targets.unsqueeze(1).transpose(-2, -1)
            else:
                # Default: use audio
                inputs = noisy_audio.to(self.device)
                targets = clean_audio.to(self.device)
                
        elif len(batch) == 2:  # Simple format: (inputs, targets)
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        
        return inputs, targets
    
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
            # Default behavior for UNet and other models
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
            # Prepare batch based on model type
            inputs, targets = self._prepare_batch(batch)
            
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
        
        # Update learning rate scheduler if available (UNet uses this)
        if self.scheduler is not None:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"   Learning rate: {current_lr:.6f}")
        
        return avg_loss
    
    def validate(self, dataloader):
        """Run validation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation")
            
            for batch in progress_bar:
                # Prepare batch based on model type
                inputs, targets = self._prepare_batch(batch)
                
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
        print(f"Model: {self.model_name}")
        print(f"Input type: {self.model_input_type}")
        
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
    
    def inference(self, input_path, output_path=None):
        """Run inference on a single audio file"""
        if hasattr(self.model, 'inference'):
            # Use model's inference method if available (like UNet wrapper)
            return self.model.inference(input_path, output_path)
        else:
            # Fallback for models without custom inference
            print("Using generic inference - consider implementing model-specific inference")
            # ... generic inference code here ...
            pass
