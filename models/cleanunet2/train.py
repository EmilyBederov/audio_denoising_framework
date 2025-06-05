# models/cleanunet2/train.py - Fixed for .pth checkpoint files

import argparse
import yaml
import torch
import torch.nn as nn
import torchaudio.transforms as T
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from dataset import load_cleanunet2_dataset
from util import print_size, LinearWarmupCosineDecay, save_checkpoint, load_checkpoint, prepare_directories_and_logger
from models import CleanUNet2
from stft_loss import MultiResolutionSTFTLoss, CleanUNet2Loss
import os
import random
import numpy as np
import glob
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def find_latest_checkpoint_pth(checkpoint_dir):
    """Find the latest .pth checkpoint file with proper epoch extraction"""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None, 0
    
    print(f" Searching for checkpoints in: {checkpoint_dir}")
    
    # Look for different .pth naming patterns
    patterns = [
        "*.pth",
        "*epoch*.pth", 
        "*_epoch_*.pth",
        "cleanunet2_epoch_*.pth",
        "checkpoint_epoch_*.pth"
    ]
    
    all_checkpoint_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        all_checkpoint_files.extend(files)
    
    # Remove duplicates
    checkpoint_files = list(set(all_checkpoint_files))
    
    if not checkpoint_files:
        print(f" No .pth checkpoint files found in: {checkpoint_dir}")
        print("Available files:")
        for f in os.listdir(checkpoint_dir):
            print(f"   {f}")
        return None, 0
    
    print(f"📂 Found {len(checkpoint_files)} checkpoint files:")
    for f in sorted(checkpoint_files):
        print(f"   {os.path.basename(f)}")
    
    # Extract epoch/iteration numbers and find the latest
    latest_file = None
    latest_number = 0
    
    for f in checkpoint_files:
        filename = os.path.basename(f)
        
        # Try different patterns to extract numbers
        number = 0
        
        # Pattern 1: cleanunet2_epoch_90.pth
        if "epoch_" in filename:
            try:
                parts = filename.split("epoch_")
                if len(parts) > 1:
                    number_str = parts[1].split(".")[0]  # Remove .pth
                    number = int(number_str)
            except (ValueError, IndexError):
                pass
        
        # Pattern 2: checkpoint_90.pth or 90.pth
        elif filename.replace(".pth", "").isdigit():
            try:
                number = int(filename.replace(".pth", ""))
            except ValueError:
                pass
                
        # Pattern 3: model_90.pth, checkpoint_90.pth etc
        else:
            try:
                # Extract last number from filename
                import re
                numbers = re.findall(r'\d+', filename)
                if numbers:
                    number = int(numbers[-1])  # Take the last number found
            except (ValueError, IndexError):
                pass
        
        if number > latest_number:
            latest_number = number
            latest_file = f
    
    if latest_file:
        print(f" Latest checkpoint: {os.path.basename(latest_file)} (epoch/iter: {latest_number})")
    
    return latest_file, latest_number

def load_checkpoint_pth(checkpoint_path, model, optimizer):
    """Load .pth checkpoint and return epoch/iteration info"""
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Format 1: Full training checkpoint with optimizer
            if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                epoch = checkpoint.get('epoch', 0)
                iteration = checkpoint.get('iteration', checkpoint.get('global_step', 0))
                learning_rate = checkpoint.get('learning_rate', optimizer.param_groups[0]['lr'])
                
                print(f"      Loaded full checkpoint:")
                print(f"      Epoch: {epoch}")
                print(f"      Iteration/Global Step: {iteration}")
                print(f"      Learning Rate: {learning_rate}")
                
                return model, optimizer, learning_rate, iteration, epoch
            
            # Format 2: Model state dict only
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                
                epoch = checkpoint.get('epoch', 0)
                iteration = checkpoint.get('iteration', checkpoint.get('global_step', 0))
                
                print(f"      Loaded model state only (no optimizer state)")
                print(f"      Epoch: {epoch}")
                print(f"      Iteration: {iteration}")
                
                return model, optimizer, optimizer.param_groups[0]['lr'], iteration, epoch
            
            # Format 3: Direct state dict
            else:
                model.load_state_dict(checkpoint)
                print(f"     Loaded direct state dict (no training info)")
                
                # Try to extract epoch from filename
                filename = os.path.basename(checkpoint_path)
                epoch = 0
                if "epoch_" in filename:
                    try:
                        epoch = int(filename.split("epoch_")[1].split(".")[0])
                    except (ValueError, IndexError):
                        pass
                
                return model, optimizer, optimizer.param_groups[0]['lr'], 0, epoch
        
        else:
            # Direct model state dict
            model.load_state_dict(checkpoint)
            print(f"    Loaded direct model weights")
            return model, optimizer, optimizer.param_groups[0]['lr'], 0, 0
            
    except Exception as e:
        print(f" Error loading checkpoint: {e}")
        raise

def train(num_gpus, rank, group_name, exp_path, checkpoint_path, log, optimization, testloader, loss_config, device=None):
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create tensorboard logger
    output_dir = os.path.join(log["directory"], exp_path)
    log_directory = os.path.join(output_dir, 'logs')
    ckpt_dir = os.path.join(output_dir, 'checkpoint')
    logger = prepare_directories_and_logger(output_dir, log_directory, ckpt_dir, rank=0)

    # Distributed initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    # Load dataset
    trainloader = load_cleanunet2_dataset(
        csv_path=trainset_config['csv_path'],
        sample_rate=config['sample_rate'],
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        power=1.0,
        crop_length_sec=0.0,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    print('Data loaded')

    # Initialize model
    model = CleanUNet2(**network_config).to(device)
    model.train()

    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=optimization["learning_rate"], weight_decay=optimization["weight_decay"])

    # Load checkpoint with .pth support
    global_step = 0
    start_epoch = 0
    steps_per_epoch = len(trainloader)
    total_epochs = config["epochs"]
    
    print(f" Training Info:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total epochs: {total_epochs}")

    if checkpoint_path is not None:
        # Check if it's a directory or specific file
        if os.path.isdir(checkpoint_path):
            latest_ckpt, latest_number = find_latest_checkpoint_pth(checkpoint_path)
        elif os.path.isfile(checkpoint_path):
            latest_ckpt = checkpoint_path
            latest_number = 0
        else:
            print(f" Checkpoint path does not exist: {checkpoint_path}")
            latest_ckpt = None
            
        if latest_ckpt:
            try:
                model, optimizer, learning_rate, iteration, epoch = load_checkpoint_pth(latest_ckpt, model, optimizer)
                
                # Use the most reliable information available
                if iteration > 0:
                    global_step = iteration
                    start_epoch = global_step // steps_per_epoch
                elif epoch > 0:
                    start_epoch = epoch
                    global_step = start_epoch * steps_per_epoch
                
                print(f" Resuming training:")
                print(f"   Start epoch: {start_epoch + 1}/{total_epochs}")
                print(f"   Global step: {global_step}")
                print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
                
            except Exception as e:
                print(f" Failed to load checkpoint: {e}")
                print("Starting from scratch...")
                start_epoch = 0
                global_step = 0
        else:
            print("No checkpoint found, starting from scratch")
    else:
        print("No checkpoint path specified, starting from scratch")

    print_size(model)

    # Define learning rate scheduler
    scheduler = LinearWarmupCosineDecay(
        optimizer,
        lr_max=optimization["learning_rate"],
        n_iter=optimization["n_iters"],
        iteration=global_step,
        divider=25,
        warmup_proportion=0.01,
        phase=('linear', 'cosine'),
    )

    # Define loss
    mrstftloss = MultiResolutionSTFTLoss(**loss_config["stft_config"]).to(device) if loss_config["stft_lambda"] > 0 else None
    loss_fn = CleanUNet2Loss(**loss_config, mrstftloss=mrstftloss)

    print(f" Starting training from epoch {start_epoch + 1}/{total_epochs} (global step {global_step})...")
    
    # Training loop with correct epoch counting
    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Progress bar shows correct epoch number
        progress_bar = tqdm(
            enumerate(trainloader), 
            total=len(trainloader),
            desc=f"Epoch {epoch+1}/{total_epochs}",
            leave=True
        )
        
        for step, (clean_audio, clean_spec, noisy_audio, noisy_spec) in progress_bar:
            noisy_audio = noisy_audio.to(device)
            noisy_spec = noisy_spec.to(device)
            clean_audio = clean_audio.to(device)
            clean_spec = clean_spec.to(device)

            optimizer.zero_grad()
            denoised_audio, denoised_spec = model(noisy_audio, noisy_spec)
            loss = loss_fn(clean_audio, denoised_audio)
            reduced_loss = reduce_tensor(loss.data, num_gpus).item() if num_gpus > 1 else loss.item()

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), optimization["max_norm"])
            scheduler.step()
            optimizer.step()

            epoch_loss += reduced_loss

            # Progress bar shows current epoch and global step
            progress_bar.set_postfix({
                "loss": f"{reduced_loss:.4f}",
                "avg_loss": f"{epoch_loss/(step+1):.4f}",
                "global_step": global_step,
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

            # Logging
            if global_step > 0 and global_step % 10 == 0:
                logger.add_scalar("Train/Train-Loss", reduced_loss, global_step)
                logger.add_scalar("Train/Gradient-Norm", grad_norm, global_step)
                logger.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], global_step)

            # Save checkpoint every N iterations
            if global_step > 0 and global_step % log["iters_per_ckpt"] == 0 and rank == 0:
                checkpoint_name = f"cleanunet2_epoch_{epoch+1}_step_{global_step}.pth"
                checkpoint_file = os.path.join(ckpt_dir, checkpoint_name)
                
                # Save in .pth format with full training state
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'loss': reduced_loss,
                    'config': config
                }, checkpoint_file)
                
                tqdm.write(f" Checkpoint saved: {checkpoint_name} (epoch {epoch+1})")

            global_step += 1

        # End of epoch summary
        avg_epoch_loss = epoch_loss / len(trainloader)
        print(f"   Epoch {epoch+1}/{total_epochs} completed")
        print(f"   Average loss: {avg_epoch_loss:.6f}")
        print(f"   Global step: {global_step}")
        print(f"   Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save epoch checkpoint
        if rank == 0:
            epoch_checkpoint = os.path.join(ckpt_dir, f"cleanunet2_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'loss': avg_epoch_loss,
                'config': config
            }, epoch_checkpoint)
            print(f"    Epoch checkpoint: {os.path.basename(epoch_checkpoint)}")

        # Validation after each epoch
        if rank == 0:
            print(f"Running validation for epoch {epoch+1}...")
            validate(model, testloader, loss_fn, global_step, trainset_config, logger, device)

    # Save final model
    if rank == 0:
        final_model_path = os.path.join(ckpt_dir, 'cleanunet2_final.pth')
        torch.save({
            'epoch': total_epochs-1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'config': config
        }, final_model_path)
        print(f" Training completed! Final model saved: {final_model_path}")

    return 0

def validate(model, val_loader, loss_fn, iteration, trainset_config, logger, device):
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", leave=False)
        for i, (clean_audio, clean_spec, noisy_audio, noisy_spec) in progress_bar:
            clean_audio, clean_spec = clean_audio.to(device), clean_spec.to(device)
            noisy_audio, noisy_spec = noisy_audio.to(device), noisy_spec.to(device)

            denoised_audio, denoised_spec = model(noisy_audio, noisy_spec)
            loss = loss_fn(clean_audio, denoised_audio)
            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
            val_loss += loss.item()
            num_batches += 1

        val_loss /= num_batches

    model.train()
    
    if logger is not None:
        print(f"Validation loss: {val_loss:.6f}")
        logger.add_scalar("Validation/Loss", val_loss, iteration)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/configs-cleanunet2/cleanunet2-config.yaml', 
                        help='YAML file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    global dist_config, network_config, trainset_config
    train_config = config.get("train_config", {})
    dist_config = config.get("dist_config", {})
    network_config = config["network_config"]
    trainset_config = config["trainset"]
    trainset_config["sample_rate"] = config["sample_rate"]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and args.group_name == '':
        print("WARNING: Multiple GPUs detected but no distributed group set")
        print("Only running 1 GPU. Use distributed.py for multiple GPUs")
        num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    testloader = load_cleanunet2_dataset(
        csv_path=config['valset']['csv_path'],
        sample_rate=config['sample_rate'],
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        power=1.0,
        crop_length_sec=0.0,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    print('Validation set loaded')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(
        num_gpus,
        args.rank,
        args.group_name,
        train_config.get("exp_path", "cleanunet2"),
        train_config.get("checkpoint_path", None),
        train_config.get("log", {}),
        train_config.get("optimization", {}),
        testloader=testloader,
        loss_config=train_config.get("loss_config", {}),
        device=device
    )
