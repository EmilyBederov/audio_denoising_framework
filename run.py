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
from core.data_loader import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='Audio Denoising Framework')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to config file')
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
    return parser.parse_args()

def main():
    print("Starting main function...")
    
    # Parse arguments
    print("Parsing arguments...")
    args = parse_args()
    print(f"Arguments parsed: config={args.config}, model={args.model}, mode={args.mode}")
    
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
    
    # Add model name to config
    print("Adding model name to config...")
    config['model_name'] = args.model
    
    print("Creating training manager...")
    # Create training manager
    manager = TrainingManager(args.model, config, device=args.device)
    print("Training manager created successfully")
    
    # Load checkpoint if provided
    if args.checkpoint:
        manager.model.load_checkpoint(args.checkpoint)
    
    # Run in the specified mode
    if args.mode == 'train':
        # Get dataloaders
        train_dataloader = get_dataloader(config, split='train')
        val_dataloader = get_dataloader(config, split='val')
        
        # Train the model
        manager.train(train_dataloader, val_dataloader)
        
    elif args.mode == 'eval':
        # Get dataloader
        test_dataloader = get_dataloader(config, split='test')
        
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
