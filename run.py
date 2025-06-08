# run.py
import argparse
import yaml
import torch
import os
from pathlib import Path
import sys

# Remove the CleanUNet2 path since it doesn't exist in your structure
project_root = Path(__file__).resolve().parent
# cleanunet2_path = project_root / "CleanUNet2"  # Removed this line
# sys.path.append(str(cleanunet2_path))  # Removed this line

from core.model_factory import ModelFactory
from core.training_manager import TrainingManager
# FIXED: Use original CleanUNet2 data loading instead of new framework
from models.cleanunet2.dataset import load_cleanunet2_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Audio Denoising Framework')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to config file')
    # FIXED: Add missing --mode argument
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'eval', 'inference'],
                        help='Running mode')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (cleanunet2, unet, ftcrngan, etc.)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--input', type=str, default=None,
                        help='Input audio file for inference')
    parser.add_argument('--output', type=str, default=None,
                        help='Output audio file for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size from config')
    return parser.parse_args()

def main():
    print("Starting main function...")
    
    # Parse arguments
    print("Parsing arguments...")
    args = parse_args()
    print(f"Arguments parsed: config={args.config}, model={args.model}, mode={args.mode}")
    
    # ADD: Early device check
    print("\n=== DEVICE SETUP ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    print("==================\n")
    
    # Check if config file exists
    print(f"Checking if config file exists: {args.config}")
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return
    print("Config file exists")
    
    # Load config
    print("Loading config...")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print(f"Config file is empty or invalid: {args.config}")
            return
            
        print(f"Loaded config from: {args.config}")
        print(f"Config keys: {list(config.keys())}")
        
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        return
    
    # Override batch size if provided
    if args.batch_size is not None:
        print(f"Overriding batch_size: {config.get('batch_size', 'not set')} -> {args.batch_size}")
        config['batch_size'] = args.batch_size
    
    # Add model name to config
    print("Adding model name to config...")
    config['model_name'] = args.model
    
    print("Creating training manager...")
    # Create training manager - PASS device explicitly
    manager = TrainingManager(args.model, config, device=str(device))
    print("Training manager created successfully")
    
    # Load checkpoint if provided
    if args.checkpoint:
        manager.model.load_checkpoint(args.checkpoint)
    
    # Run in the specified mode
    if args.mode == 'train':
        # FIXED: Use original CleanUNet2 data loading (same as SLURM training)
        print("Creating dataloaders (original CleanUNet2 method)...")
        
        train_dataloader = load_cleanunet2_dataset(
            csv_path=config['trainset']['csv_path'],
            sample_rate=config['sample_rate'],
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=1.0,
            crop_length_sec=0.0,  # NO CROPPING - same as SLURM training!
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        val_dataloader = load_cleanunet2_dataset(
            csv_path=config['valset']['csv_path'],
            sample_rate=config['sample_rate'],
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=1.0,
            crop_length_sec=0.0,  # NO CROPPING - same as SLURM training!
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        # Show dataloader info
        train_steps = len(train_dataloader)
        val_steps = len(val_dataloader)
        print(f"Training steps: {train_steps:,} (should be ~5,191)")
        print(f"Validation steps: {val_steps:,} (should be ~577)")
        
        if abs(train_steps - 5191) < 100:
            print("✅ SUCCESS! Training steps match SLURM training")
        else:
            print(f"⚠️ Warning: Training steps ({train_steps}) differ from SLURM (5,191)")
        
        # Train the model
        manager.train(train_dataloader, val_dataloader)
        
    elif args.mode == 'eval':
        # Use original data loading for evaluation too
        test_dataloader = load_cleanunet2_dataset(
            csv_path=config['valset']['csv_path'],
            sample_rate=config['sample_rate'],
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=1.0,
            crop_length_sec=0.0,  # NO CROPPING
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        # Evaluate the model
        loss, metrics = manager.evaluate(test_dataloader)
        
        # Print results
        print(f"Test loss: {loss:.4f}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
    elif args.mode == 'inference':
        # Check if input file is provided
        if not args.input:
            print("Error: Input audio file is required for inference mode")
            return
        
        # Run inference
        output_path = manager.inference(args.input, args.output)
        print(f"Denoised audio saved to: {output_path}")

if __name__ == "__main__":
    main()
