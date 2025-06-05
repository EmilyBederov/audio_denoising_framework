# models/cleanunet2/train_fixed.py - Fixed to properly resume training from correct epoch

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

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file with proper epoch extraction"""
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.pkl'))
    if not checkpoint_files:
        return None, 0
    
    # Extract iteration numbers and find the latest
    latest_file = None
    latest_iteration = 0
    
    for f in checkpoint_files:
        try:
            filename = os.path.basename(f)
            if filename.endswith('.pkl'):
                iteration = int(filename[:-4])  # Remove .pkl extension
                if iteration > latest_iteration:
                    latest_iteration = iteration
                    latest_file = f
        except ValueError:
            continue
    
    return latest_file, latest_iteration

def calculate_epoch_from_iteration(iteration, steps_per_epoch):
    """Calculate epoch number from iteration"""
    return iteration // steps_per_epoch

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

    # Load checkpoint with proper epoch calculation
    global_step = 0
    start_epoch = 0
    steps_per_epoch = len(trainloader)
    total_epochs = config["epochs"]
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total epochs: {total_epochs}")

    if checkpoint_path is not None:
        latest_ckpt, latest_iteration = find_latest_checkpoint(checkpoint_path)
        if latest_ckpt:
            print(f"Resuming from latest checkpoint: {latest_ckpt}")
            model, optimizer, learning_rate, iteration = load_checkpoint(latest_ckpt, model, optimizer)
            global_step = iteration
            start_epoch = calculate_epoch_from_iteration(global_step, steps_per_epoch)
            print(f"✅ Resuming from:")
            print(f"   Global step: {global_step}")
            print(f"   Epoch: {start_epoch + 1}/{total_epochs}")
            print(f"   Learning rate: {learning_rate}")
        else:
            print("No checkpoint found, starting from scratch")
    else:
        print("No checkpoint directory specified, starting from scratch")

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

    print(f"🚀 Starting training from epoch {start_epoch + 1}/{total_epochs} (global step {global_step})...")
    
    # FIXED: Training loop with correct epoch counting and progress tracking
    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Calculate steps remaining in current epoch if resuming mid-epoch
        epoch_start_step = epoch * steps_per_epoch
        steps_completed_in_epoch = max(0, global_step - epoch_start_step)
        
        # FIXED: Progress bar shows correct epoch number
        progress_bar = tqdm(
            enumerate(trainloader), 
            total=len(trainloader),
            desc=f"Epoch {epoch+1}/{total_epochs}",
            initial=steps_completed_in_epoch,  # Resume from correct step in epoch
            leave=True
        )
        
        for step, (clean_audio, clean_spec, noisy_audio, noisy_spec) in progress_bar:
            # Skip steps we've already completed if resuming mid-epoch
            if global_step < epoch_start_step + step:
                global_step += 1
                continue
                
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

            # FIXED: Progress bar shows current step within epoch and global step
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

            # Save checkpoint
            if global_step > 0 and global_step % log["iters_per_ckpt"] == 0 and rank == 0:
                checkpoint_name = f"{global_step}.pkl"
                checkpoint_file = os.path.join(ckpt_dir, checkpoint_name)
                save_checkpoint(model, optimizer, optimization["learning_rate"], global_step, checkpoint_file)
                tqdm.write(f"[✓] Checkpoint saved: {checkpoint_name} (epoch {epoch+1})")

            global_step += 1

        # End of epoch summary
        avg_epoch_loss = epoch_loss / len(trainloader)
        print(f"✅ Epoch {epoch+1}/{total_epochs} completed")
        print(f"   Average loss: {avg_epoch_loss:.6f}")
        print(f"   Global step: {global_step}")
        print(f"   Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation after each epoch
        if rank == 0:
            print(f"Running validation for epoch {epoch+1}...")
            validate(model, testloader, loss_fn, global_step, trainset_config, logger, device)

    # Save final model
    if rank == 0:
        final_model_path = os.path.join(ckpt_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"[✅] Training completed! Final model saved: {final_model_path}")

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
        train_config["exp_path"],
        train_config["checkpoint_path"],
        train_config["log"],
        train_config["optimization"],
        testloader=testloader,
        loss_config=train_config.get("loss_config", {}),
        device=device
    )
