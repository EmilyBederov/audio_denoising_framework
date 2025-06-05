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

def check_for_nan_and_inf(tensor, tensor_name="tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"{tensor_name} has NaN values!")
    if torch.isinf(tensor).any():
        raise ValueError(f"{tensor_name} has Inf values!")

def validate(model, val_loader, loss_fn, iteration, trainset_config, logger, device):
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating", leave=False)
        for i, (clean_audio, clean_spec, noisy_audio, noisy_spec) in progress_bar:
            clean_audio, clean_spec = clean_audio.to(device), clean_spec.to(device)
            noisy_audio, noisy_spec = noisy_audio.to(device), noisy_spec.to(device)

            denoised_audio, denoised_spec = model(noisy_audio, noisy_spec)
            loss = loss_fn(clean_audio, denoised_audio)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            val_loss += loss.item()
            num_batches += 1

        val_loss /= num_batches

    model.train()

    mel_transform = T.MelSpectrogram(
        sample_rate=trainset_config['sample_rate'],
        n_fft=trainset_config['n_fft'],
        win_length=trainset_config['win_length'],
        hop_length=trainset_config['hop_length'],
        n_mels=80
    ).to(device)
    amplitude_to_db = T.AmplitudeToDB(stype='power')

    if logger is not None:
        print(f"Validation loss at iteration {iteration}: {val_loss:.6f}")
        logger.add_scalar("Validation/Loss", val_loss, iteration)

        num_samples = min(4, clean_spec.size(0))
        for i in range(num_samples):
            clean_audio_i = clean_audio[i].squeeze()
            denoised_audio_i = denoised_audio[i].squeeze()
            noisy_audio_i = noisy_audio[i].squeeze()

            clean_audio_np = clean_audio_i.cpu().numpy()
            denoised_audio_np = denoised_audio_i.cpu().numpy()
            noisy_audio_np = noisy_audio_i.cpu().numpy()

            clean_spec = amplitude_to_db(mel_transform(clean_audio_i))
            denoised_spec = amplitude_to_db(mel_transform(denoised_audio_i))
            noisy_spec = amplitude_to_db(mel_transform(noisy_audio_i))

            clean_spec_np = clean_spec.cpu().numpy()
            denoised_spec_np = denoised_spec.cpu().numpy()
            noisy_spec_np = noisy_spec.cpu().numpy()

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(clean_spec_np, origin='lower', aspect='auto')
            axs[0].set_title('Clean Spectrogram')
            axs[1].imshow(denoised_spec_np, origin='lower', aspect='auto')
            axs[1].set_title('Denoised Spectrogram')
            axs[2].imshow(noisy_spec_np, origin='lower', aspect='auto')
            axs[2].set_title('Noisy Spectrogram')
            plt.tight_layout()
            logger.add_figure(f'Spectrograms/Sample_{i}', fig, iteration)
            plt.close(fig)

            sample_rate = trainset_config['sample_rate']
            logger.add_audio(f'Audio/Clean_{i}', clean_audio_np, iteration, sample_rate=sample_rate)
            logger.add_audio(f'Audio/Denoised_{i}', denoised_audio_np, iteration, sample_rate=sample_rate)
            logger.add_audio(f'Audio/Noisy_{i}', noisy_audio_np, iteration, sample_rate=sample_rate)

def train(num_gpus, rank, group_name, exp_path, checkpoint_path, checkpoint_cleanunet_path, checkpoint_cleanspecnet_path, log, optimization, testloader, freeze_cleanspecnet=False, freeze_cleanunet=False, loss_config=None, device=None):
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

    # Load checkpoint
    global_step = 0
    start_epoch = 0
    steps_per_epoch = len(trainloader)
    total_epochs = config["epochs"]
    latest_ckpt = None

    if checkpoint_path is not None and os.path.isdir(checkpoint_path):
        ckpt_files = glob.glob(os.path.join(checkpoint_path, '*.pkl'))
        if ckpt_files:
            latest_ckpt = max(ckpt_files, key=os.path.getctime)
            print(f"Resuming from latest checkpoint: {latest_ckpt}")
            model, optimizer, learning_rate, iteration = load_checkpoint(latest_ckpt, model, optimizer)
            global_step = iteration
            start_epoch = global_step // steps_per_epoch
            print(f"Resuming from epoch {start_epoch + 1}, global step {global_step}")
        else:
            print(f"No checkpoint files found in: {checkpoint_path}")
            print("Starting training from scratch.")
    else:
        print(f"No checkpoint directory found: {checkpoint_path}")
        print("Starting training from scratch.")

    # Load additional checkpoints if provided
    if checkpoint_cleanunet_path is not None and os.path.exists(checkpoint_cleanunet_path):
        print(f"Loading checkpoint '{checkpoint_cleanunet_path}'")
        checkpoint_dict = torch.load(checkpoint_cleanunet_path, map_location='cpu')
        new_checkpoint_dict = {f"clean_unet.{k}": v for k, v in checkpoint_dict['model_state_dict'].items()}
        model.load_state_dict(new_checkpoint_dict, strict=False)
    if checkpoint_cleanspecnet_path is not None and os.path.exists(checkpoint_cleanspecnet_path):
        print(f"Loading checkpoint '{checkpoint_cleanspecnet_path}'")
        checkpoint_dict = torch.load(checkpoint_cleanspecnet_path, map_location='cpu')
        new_checkpoint_dict = {f"clean_spec_net.{k}": v for k, v in checkpoint_dict['state_dict'].items()}
        model.load_state_dict(new_checkpoint_dict, strict=False)

    if freeze_cleanspecnet:
        for param in model.clean_spec_net.parameters():
            param.requires_grad = False
    if freeze_cleanunet:
        for param in model.clean_unet.parameters():
            param.requires_grad = False

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

    print(f"Starting training from epoch {start_epoch + 1}/{total_epochs}...")
    for epoch in tqdm(range(start_epoch, total_epochs), desc="Epoch", initial=start_epoch, total=total_epochs, position=0):
        model.train()
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{total_epochs}", position=1, leave=False)
        for step, (clean_audio, clean_spec, noisy_audio, noisy_spec) in progress_bar:
            noisy_audio = noisy_audio.to(device)
            noisy_spec = noisy_spec.to(device)
            clean_audio = clean_audio.to(device)
            clean_spec = clean_spec.to(device)

            try:
                check_for_nan_and_inf(noisy_audio, "noisy_audio")
                check_for_nan_and_inf(noisy_spec, "noisy_spec")
                check_for_nan_and_inf(clean_audio, "clean_audio")
                check_for_nan_and_inf(clean_spec, "clean_spec")
            except ValueError as e:
                tqdm.write(str(e))
                continue

            optimizer.zero_grad()
            denoised_audio, denoised_spec = model(noisy_audio, noisy_spec)
            loss = loss_fn(clean_audio, denoised_audio)
            reduced_loss = reduce_tensor(loss.data, num_gpus).item() if num_gpus > 1 else loss.item()

            loss.backward()
            if torch.isnan(loss).any():
                tqdm.write("Loss contains NaN, skipping step")
                continue

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), optimization["max_norm"])
            scheduler.step()
            optimizer.step()

            progress_bar.set_postfix({
                "step": step,
                "global_step": global_step,
                "loss": f"{reduced_loss:.4f}"
            })

            if global_step > 0 and global_step % 10 == 0:
                logger.add_scalar("Train/Train-Loss", reduced_loss, global_step)
                logger.add_scalar("Train/Gradient-Norm", grad_norm, global_step)
                logger.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], global_step)

            if global_step > 0 and global_step % log["iters_per_ckpt"] == 0 and rank == 0:
                checkpoint_name = f"{global_step}.pkl"
                checkpoint_file = os.path.join(ckpt_dir, checkpoint_name)
                save_checkpoint(model, optimizer, optimization["learning_rate"], global_step, checkpoint_file)
                tqdm.write(f"[✓] Checkpoint saved at: {checkpoint_name}")

            global_step += 1

        # Validation after each epoch
        if rank == 0:
            print(f"Epoch {epoch + 1} finished. Running validation...")
            model.eval()
            validate(model, testloader, loss_fn, global_step, trainset_config, logger, device)
            model.train()

    # Save final model
    if rank == 0:
        final_model_path = os.path.join(ckpt_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"[✓] Final model weights saved for inference: {final_model_path}")

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/configs-cleanunet2/cleanunet2-config.yaml', 
                        help='YAML file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0, help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='', help='name of group for distributed')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_config = config["train_config"]
    global dist_config
    dist_config = config["dist_config"]
    global network_config
    network_config = config["network_config"]
    global trainset_config
    trainset_config = config["trainset"]
    trainset_config["sample_rate"] = config["sample_rate"]
    trainset_config["n_fft"] = 1024
    trainset_config["win_length"] = 1024
    trainset_config["hop_length"] = 256

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
        train_config["checkpoint_cleanunet_path"],
        train_config["checkpoint_cleanspecnet_path"],
        train_config["log"],
        train_config["optimization"],
        testloader=testloader,
        freeze_cleanspecnet=train_config.get("freeze_cleanspecnet", False),
        freeze_cleanunet=train_config.get("freeze_cleanunet", False),
        loss_config=train_config.get("loss_config", None),
        device=device
    )  
    print("noisy_spec shape:", noisy_spec.shape)

    
