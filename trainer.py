cat > ~/work/audio_denoising_framework/core/trainer.py << 'EOF'
# Unified Trainer class
import torch
import torch.optim as optim
import os
import logging
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataloader, config, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Setup optimizer based on config
        self.optimizer = self._get_optimizer()
        
        # Setup learning rate scheduler if specified
        self.scheduler = self._get_scheduler()
        
        # Setup logging
        self.logger = logging.getLogger("trainer")
        
    def _get_optimizer(self):
        """Initialize optimizer based on config"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adam')
        lr = opt_config.get('lr', 0.001)
        
        if opt_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif opt_type.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr)
        # Add more optimizers as needed
        
    def _get_scheduler(self):
        """Initialize LR scheduler if specified in config"""
        scheduler_config = self.config.get('scheduler', None)
        if not scheduler_config:
            return None
            
        # Implement various schedulers based on config
        return None  # Placeholder
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.dataloader['train'])
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.model.get_loss(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_description(f"Epoch {epoch} - Loss: {loss.item():.4f}")
            
        # Update scheduler if exists
        if self.scheduler:
            self.scheduler.step()
            
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.dataloader['train'])
        return avg_loss
    
    def validate(self):
        """Run validation"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in self.dataloader['val']:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.model.get_loss(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
        
        # Calculate validation metrics
        avg_loss = total_loss / len(self.dataloader['val'])
        return avg_loss
    
    def train(self, num_epochs):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Log metrics
            self.logger.info(f"Epoch {epoch}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"best_model.pth")
                
            # Regular checkpoint saving
            if epoch % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'))
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
EOF
