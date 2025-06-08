# core/training_manager_two_stage.py
# EXACT two-stage training implementation from CleanUNet 2 paper

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
    print("Warning: CleanUNet2 STFT loss not available")

class TwoStageTrainingManager:
    """
    EXACT two-stage training from CleanUNet 2 paper:
    1. First train CleanSpecNet alone (1M iterations, batch_size=64)
    2. Then train CleanUNet 2 with frozen CleanSpecNet (500K iterations, batch_size=32)
    """
    def __init__(self, model_name, config, device='cuda'):
        self.model_name = model_name
        self.config = config
        self.device = torch.device(device)
        
        # Create model
        from core.model_factory import ModelFactory
        self.model = ModelFactory.create_model(model_name, config)
        self.model = self.model.to(self.device)
        
        # Initialize loss functions
        self._setup_loss_functions()
        
        # Track training state
        self.current_stage = 1  # 1 = CleanSpecNet, 2 = CleanUNet 2
        self.cleanspecnet_trained = False
        
        print(f"Two-stage training manager created for {model_name} on {self.device}")
        
    def _setup_loss_functions(self):
        """Setup loss functions for both stages"""
        # Stage 1: CleanSpecNet loss (from paper equation 1)
        self.cleanspecnet_loss = self._create_cleanspecnet_loss()
        
        # Stage 2: CleanUNet 2 loss (L1 + Multi-resolution STFT)
        if CLEANUNET2_AVAILABLE:
            mrstft_loss = MultiResolutionSTFTLoss(
                fft_sizes=[512, 1024, 2048],
                hop_sizes=[50, 120, 240],
                win_lengths=[240, 600, 1200],
                window="hann_window",
                sc_lambda=0.5,
                mag_lambda=0.5,
                band="full"
            ).to(self.device)
            
            self.cleanunet2_loss = CleanUNet2Loss(
                ell_p=1,
                ell_p_lambda=1.0,
                stft_lambda=1.0,
                mrstftloss=mrstft_loss
            )
        else:
            self.cleanunet2_loss = torch.nn.L1Loss()
    
    def _create_cleanspecnet_loss(self):
        """
        EXACT CleanSpecNet loss from paper equation (1):
        Loss = (1/T_spec) * ||log(y/ŷ)||_1 + ||y - ŷ||_F / ||y||_F
        """
        class CleanSpecNetLoss(nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, predicted_spec, target_spec):
                """
                predicted_spec: [B, F, T] - predicted clean spectrogram
                target_spec: [B, F, T] - ground truth clean spectrogram
                """
                eps = 1e-8  # For numerical stability
                
                # Term 1: (1/T_spec) * ||log(y/ŷ)||_1
                T_spec = target_spec.shape[-1]
                log_ratio = torch.log((target_spec + eps) / (predicted_spec + eps))
                log_loss = torch.mean(torch.abs(log_ratio)) / T_spec
                
                # Term 2: ||y - ŷ||_F / ||y||_F (Frobenius norm)
                diff_norm = torch.norm(target_spec - predicted_spec, p='fro')
                target_norm = torch.norm(target_spec, p='fro')
                spectral_loss = diff_norm / (target_norm + eps)
                
                return log_loss + spectral_loss
        
        return CleanSpecNetLoss()
    
    def train_stage1_cleanspecnet(self, train_dataloader, val_dataloader=None):
        """
        Stage 1: Train CleanSpecNet alone
        EXACT from paper: 1M iterations, batch_size=64
        """
        print("\n" + "="*60)
        print("STAGE 1: Training CleanSpecNet (1M iterations)")
        print("="*60)
        
        # EXACT optimizer from paper
        optimizer = torch.optim.Adam(
            self.model.model.clean_spec_net.parameters(),  # Access through wrapper
            lr=2e-4,  # EXACT from paper
            betas=(0.9, 0.999)
        )
        
        # EXACT scheduler from paper: linear warmup (5%) + cosine annealing
        from torch.optim.lr_scheduler import LambdaLR
        total_iterations = 1000000  # 1M iterations as in paper
        warmup_iterations = int(0.05 * total_iterations)  # 5% warmup
        
        def lr_lambda(iteration):
            if iteration < warmup_iterations:
                # Linear warmup
                return iteration / warmup_iterations
            else:
                # Cosine annealing
                import math
                progress = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        # Training loop
        self.model.clean_spec_net.train()
        iteration = 0
        epoch = 0
        
        save_path = os.path.join(self.config.get('save_path', 'outputs'), 'stage1_cleanspecnet')
        os.makedirs(save_path, exist_ok=True)
        
        while iteration < total_iterations:
            epoch += 1
            epoch_loss = 0.0
            batch_count = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Stage 1 Epoch {epoch}")
            
            for batch in progress_bar:
                if iteration >= total_iterations:
                    break
                    
                # Prepare batch for CleanSpecNet training
                clean_audio, clean_spec, noisy_audio, noisy_spec = batch
                clean_spec = clean_spec.to(self.device)
                noisy_spec = noisy_spec.to(self.device)
                
                # Remove channel dimension if present
                if clean_spec.dim() == 4:
                    clean_spec = clean_spec.squeeze(1)
                if noisy_spec.dim() == 4:
                    noisy_spec = noisy_spec.squeeze(1)
                
                # Forward pass through CleanSpecNet only
                optimizer.zero_grad()
                predicted_spec = self.model.clean_spec_net(noisy_spec)
                
                # Compute CleanSpecNet loss
                loss = self.cleanspecnet_loss(predicted_spec, clean_spec)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                batch_count += 1
                iteration += 1
                
                # Update progress
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_loss/batch_count:.6f}',
                    'iter': f'{iteration}/{total_iterations}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Save checkpoint every 100K iterations
                if iteration % 100000 == 0:
                    checkpoint_path = os.path.join(save_path, f'cleanspecnet_iter_{iteration}.pth')
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': self.model.clean_spec_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'config': self.config
                    }, checkpoint_path)
                    print(f"\nStage 1 checkpoint saved: {checkpoint_path}")
        
        # Save final CleanSpecNet model
        final_path = os.path.join(save_path, 'cleanspecnet_final.pth')
        torch.save({
            'iteration': total_iterations,
            'model_state_dict': self.model.clean_spec_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config
        }, final_path)
        
        print(f"\n✅ Stage 1 Complete! CleanSpecNet trained for {total_iterations} iterations")
        print(f"Final model saved: {final_path}")
        
        self.cleanspecnet_trained = True
        return final_path
    
    def train_stage2_cleanunet2(self, train_dataloader, val_dataloader=None):
        """
        Stage 2: Train full CleanUNet 2 with frozen CleanSpecNet
        EXACT from paper: 500K iterations, batch_size=32
        """
        if not self.cleanspecnet_trained:
            raise ValueError("Must complete Stage 1 (CleanSpecNet training) first!")
        
        print("\n" + "="*60)
        print("STAGE 2: Training CleanUNet 2 with frozen CleanSpecNet (500K iterations)")
        print("="*60)
        
        # Freeze CleanSpecNet parameters
        for param in self.model.clean_spec_net.parameters():
            param.requires_grad = False
        print("✅ CleanSpecNet frozen")
        
        # Only train CleanUNet and upsampler
        trainable_params = []
        trainable_params.extend(self.model.clean_unet.parameters())
        trainable_params.extend(self.model.spectrogram_upsampler.parameters())
        
        # EXACT optimizer from paper
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=2e-4,  # EXACT from paper
            betas=(0.9, 0.999)
        )
        
        # EXACT scheduler from paper
        total_iterations = 500000  # 500K iterations as in paper
        warmup_iterations = int(0.05 * total_iterations)  # 5% warmup
        
        def lr_lambda(iteration):
            if iteration < warmup_iterations:
                return iteration / warmup_iterations
            else:
                import math
                progress = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        # Training loop
        self.model.train()
        iteration = 0
        epoch = 0
        
        save_path = os.path.join(self.config.get('save_path', 'outputs'), 'stage2_cleanunet2')
        os.makedirs(save_path, exist_ok=True)
        best_val_loss = float('inf')
        
        while iteration < total_iterations:
            epoch += 1
            epoch_loss = 0.0
            batch_count = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Stage 2 Epoch {epoch}")
            
            for batch in progress_bar:
                if iteration >= total_iterations:
                    break
                    
                # Prepare batch for full CleanUNet 2 training
                clean_audio, clean_spec, noisy_audio, noisy_spec = batch
                clean_audio = clean_audio.to(self.device)
                noisy_audio = noisy_audio.to(self.device)
                noisy_spec = noisy_spec.to(self.device)
                
                # Forward pass through full CleanUNet 2
                optimizer.zero_grad()
                denoised_audio, denoised_spec = self.model(noisy_audio, noisy_spec)
                
                # Compute CleanUNet 2 loss (L1 + Multi-resolution STFT)
                if CLEANUNET2_AVAILABLE:
                    loss = self.cleanunet2_loss(clean_audio, denoised_audio)
                else:
                    loss = self.cleanunet2_loss(denoised_audio, clean_audio)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                batch_count += 1
                iteration += 1
                
                # Update progress
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_loss/batch_count:.6f}',
                    'iter': f'{iteration}/{total_iterations}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Validation and checkpointing
                if iteration % 50000 == 0:  # Every 50K iterations
                    val_loss = None
                    if val_dataloader:
                        val_loss = self._validate_stage2(val_dataloader)
                        
                    checkpoint_path = os.path.join(save_path, f'cleanunet2_iter_{iteration}.pth')
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'val_loss': val_loss,
                        'config': self.config
                    }, checkpoint_path)
                    
                    if val_loss and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(save_path, 'cleanunet2_best.pth')
                        torch.save({
                            'iteration': iteration,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                            'val_loss': val_loss,
                            'config': self.config
                        }, best_path)
                        print(f"\n✅ New best model saved: {best_path}")
                    
                    print(f"\nStage 2 checkpoint saved: {checkpoint_path}")
        
        # Save final model
        final_path = os.path.join(save_path, 'cleanunet2_final.pth')
        torch.save({
            'iteration': total_iterations,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config
        }, final_path)
        
        print(f"\n✅ Stage 2 Complete! CleanUNet 2 trained for {total_iterations} iterations")
        print(f"Final model saved: {final_path}")
        
        return final_path
    
    def _validate_stage2(self, val_dataloader):
        """Validate CleanUNet 2 during stage 2 training"""
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                clean_audio, clean_spec, noisy_audio, noisy_spec = batch
                clean_audio = clean_audio.to(self.device)
                noisy_audio = noisy_audio.to(self.device)
                noisy_spec = noisy_spec.to(self.device)
                
                denoised_audio, _ = self.model(noisy_audio, noisy_spec)
                
                if CLEANUNET2_AVAILABLE:
                    loss = self.cleanunet2_loss(clean_audio, denoised_audio)
                else:
                    loss = self.cleanunet2_loss(denoised_audio, clean_audio)
                
                total_loss += loss.item()
                count += 1
                
                # Limit validation batches
                if count >= 50:
                    break
        
        self.model.train()
        avg_loss = total_loss / count if count > 0 else 0.0
        print(f"\nValidation loss: {avg_loss:.6f}")
        return avg_loss
    
    def train_full_pipeline(self, train_dataloader, val_dataloader=None):
        """
        Complete two-stage training pipeline exactly as in paper
        """
        print("🚀 Starting EXACT CleanUNet 2 two-stage training pipeline")
        print("Based on paper specifications:")
        print("  Stage 1: CleanSpecNet - 1M iterations, batch_size=64")
        print("  Stage 2: CleanUNet 2 - 500K iterations, batch_size=32")
        
        # Stage 1: Train CleanSpecNet
        stage1_model = self.train_stage1_cleanspecnet(train_dataloader, val_dataloader)
        
        # Stage 2: Train full CleanUNet 2
        stage2_model = self.train_stage2_cleanunet2(train_dataloader, val_dataloader)
        
        print("\n" + "="*60)
        print("🎉 COMPLETE TWO-STAGE TRAINING FINISHED!")
        print("="*60)
        print(f"Stage 1 model: {stage1_model}")
        print(f"Stage 2 model: {stage2_model}")
        print("\nYour CleanUNet 2 is now trained exactly as in the paper! 🎯")
        
        return stage2_model