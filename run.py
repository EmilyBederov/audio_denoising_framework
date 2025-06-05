# run.py - CORRECTED VERSION (Uses original CleanUNet2 data loading)
import argparse
import yaml
import torch
import os
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from core.model_factory import ModelFactory
from core.training_manager import TrainingManager

# FIXED: Use original CleanUNet2 data loading instead of new framework
from models.cleanunet2.dataset import load_cleanunet2_dataset

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
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint directory or file for resuming')
    return parser.parse_args()

def validate_data_setup(config):
    """Validate that data files exist and are accessible"""
    print("\n🔍 VALIDATING DATA SETUP...")
    
    train_csv = config['trainset']['csv_path']
    val_csv = config['valset']['csv_path']
    
    issues = []
    
    # Check training CSV
    if not os.path.exists(train_csv):
        issues.append(f"Training CSV not found: {train_csv}")
    else:
        try:
            import pandas as pd
            df = pd.read_csv(train_csv)
            print(f"✅ Training CSV: {len(df):,} pairs")
        except Exception as e:
            issues.append(f"Error reading training CSV: {e}")
    
    # Check validation CSV  
    if not os.path.exists(val_csv):
        issues.append(f"Validation CSV not found: {val_csv}")
    else:
        try:
            import pandas as pd
            df = pd.read_csv(val_csv)
            print(f"✅ Validation CSV: {len(df):,} pairs")
        except Exception as e:
            issues.append(f"Error reading validation CSV: {e}")
    
    if issues:
        print("❌ DATA ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease fix these issues before training!")
        return False
    
    print("✅ Data validation passed!")
    return True

def create_data_loaders_original(config):
    """Create data loaders using ORIGINAL CleanUNet2 approach (no cropping!)"""
    print("\n📊 CREATING DATA LOADERS (Original CleanUNet2 Method)...")
    
    # FIXED: Use original CleanUNet2 data loading approach
    print("Creating training dataloader (original method)...")
    train_dataloader = load_cleanunet2_dataset(
        csv_path=config['trainset']['csv_path'],
        sample_rate=config['sample_rate'],
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        power=1.0,
        crop_length_sec=0.0,  # ✅ NO CROPPING (same as SLURM training)
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    train_steps = len(train_dataloader)
    train_samples = train_steps * config['batch_size']
    
    print(f"   Training: {train_samples:,} samples, {train_steps:,} steps/epoch")
    
    # Create validation dataloader
    print("Creating validation dataloader (original method)...")
    val_dataloader = load_cleanunet2_dataset(
        csv_path=config['valset']['csv_path'],
        sample_rate=config['sample_rate'],
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        power=1.0,
        crop_length_sec=0.0,  # ✅ NO CROPPING (same as SLURM training)
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    val_steps = len(val_dataloader)
    val_samples = val_steps * config['batch_size']
    
    print(f"   Validation: {val_samples:,} samples, {val_steps:,} steps")
    
    # VALIDATION: Check if this matches SLURM training
    expected_train_steps = 5191  # From SLURM logs
    if abs(train_steps - expected_train_steps) < 100:  # Allow some tolerance
        print(f"✅ PERFECT! Training steps ({train_steps:,}) match SLURM training ({expected_train_steps:,})")
        print(f"✅ You're now using the FULL dataset again!")
    else:
        print(f"⚠️ Training steps ({train_steps:,}) differ from SLURM ({expected_train_steps:,})")
        if train_steps < expected_train_steps * 0.5:
            print(f"❌ Still only using {train_steps/expected_train_steps*100:.1f}% of training data")
        else:
            print(f"📊 Using {train_steps/expected_train_steps*100:.1f}% of SLURM data (close enough)")
    
    return train_dataloader, val_dataloader

def main():
    print("🚀 Audio Denoising Framework - CORRECTED (Original Data Loading)")
    print("=" * 70)
    
    # Parse arguments
    args = parse_args()
    print(f"Arguments: config={args.config}, model={args.model}, mode={args.mode}")
    
    # Device setup
    print(f"\n🖥️ DEVICE SETUP")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    print(f"\n📄 CONFIG LOADING")
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print(f"❌ Config file is empty or invalid: {args.config}")
            return
            
        print(f"✅ Loaded config from: {args.config}")
        print(f"Config keys: {list(config.keys())}")
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Override batch size if provided
    if args.batch_size is not None:
        print(f"🔧 Overriding batch_size: {config.get('batch_size', 'not set')} -> {args.batch_size}")
        config['batch_size'] = args.batch_size
    
    # Add model name to config
    config['model_name'] = args.model
    
    # Validate data setup before creating anything
    if args.mode == 'train':
        if not validate_data_setup(config):
            return
    
    # Create training manager
    print(f"\n🏗️ MODEL SETUP")
    try:
        manager = TrainingManager(args.model, config, device=str(device))
        print("✅ Training manager created successfully")
    except Exception as e:
        print(f"❌ Error creating training manager: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\n📦 LOADING CHECKPOINT: {args.checkpoint}")
        try:
            manager.load_checkpoint(args.checkpoint)
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return
    elif args.checkpoint_path:
        # Handle checkpoint directory or specific file
        print(f"\n📦 LOADING CHECKPOINT: {args.checkpoint_path}")
        try:
            if os.path.isdir(args.checkpoint_path):
                # Find latest checkpoint in directory
                import glob
                checkpoints = glob.glob(os.path.join(args.checkpoint_path, "*.pth"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=os.path.getctime)
                    print(f"   Found latest checkpoint: {latest_checkpoint}")
                    manager.load_checkpoint(latest_checkpoint)
                else:
                    print(f"   No .pth files found in {args.checkpoint_path}")
            else:
                # Direct file path
                manager.load_checkpoint(args.checkpoint_path)
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return
    
    # Run in the specified mode
    if args.mode == 'train':
        # FIXED: Create data loaders using original CleanUNet2 approach
        train_dataloader, val_dataloader = create_data_loaders_original(config)
        
        print(f"\n🎯 STARTING TRAINING")
        print(f"Model: {args.model}")
        print(f"Epochs: {config.get('epochs', 100)}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Training steps per epoch: {len(train_dataloader):,}")
        print(f"Validation steps: {len(val_dataloader):,}")
        print(f"Using ORIGINAL CleanUNet2 data loading (crop_length_sec=0.0)")
        print("=" * 70)
        
        # Train the model
        try:
            manager.train(train_dataloader, val_dataloader)
            print("\n🎉 Training completed successfully!")
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
        
    elif args.mode == 'eval':
        # FIXED: Use original data loading for evaluation too
        print(f"\n📊 STARTING EVALUATION")
        
        test_dataloader = load_cleanunet2_dataset(
            csv_path=config['valset']['csv_path'],  # or testset if available
            sample_rate=config['sample_rate'],
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=1.0,
            crop_length_sec=0.0,  # NO CROPPING
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        print(f"Test samples: {len(test_dataloader) * config['batch_size']:,}")
        
        # Evaluate the model
        try:
            loss, metrics = manager.evaluate(test_dataloader)
            
            # Print results
            print(f"\n📈 EVALUATION RESULTS:")
            print(f"Test loss: {loss:.4f}")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
    elif args.mode == 'inference':
        # Check if input file is provided
        if not args.input:
            print("❌ Error: Input audio file is required for inference mode")
            return
        
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file not found: {args.input}")
            return
        
        print(f"\n🎙️ RUNNING INFERENCE")
        print(f"Input: {args.input}")
        print(f"Output: {args.output or 'auto-generated'}")
        
        # Run inference
        try:
            output_path = manager.inference(args.input, args.output)
            print(f"✅ Denoised audio saved to: {output_path}")
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
