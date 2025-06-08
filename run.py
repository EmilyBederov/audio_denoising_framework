
# run_paper_exact.py
# Script to run EXACT two-stage training as specified in paper

import argparse
import yaml
import torch
import os
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from core.training_manager_two_stage import TwoStageTrainingManager
from core.data_loader import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='CleanUNet 2 - EXACT Paper Implementation')
    parser.add_argument('--config', type=str, 
                        default='configs/configs-cleanunet2/cleanunet2-paper-exact.yaml',
                        help='Path to config file')
    parser.add_argument('--stage', type=str, default='both', 
                        choices=['1', '2', 'both'],
                        help='Training stage: 1=CleanSpecNet, 2=CleanUNet2, both=complete pipeline')
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                        help='Path to stage 1 checkpoint (for stage 2 training)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    return parser.parse_args()

def main():
    print("🎯 CleanUNet 2 - EXACT Paper Implementation")
    print("=" * 60)
    
    args = parse_args()
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        return
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config: {args.config}")
    print(f"Training mode: {config.get('training_mode', 'standard')}")
    
    # Create two-stage training manager
    manager = TwoStageTrainingManager("cleanunet2", config, device=str(device))
    
    # Prepare data loaders with EXACT batch sizes from paper
    if args.stage in ['1', 'both']:
        # Stage 1: batch_size=64 (CleanSpecNet)
        config_stage1 = config.copy()
        config_stage1['batch_size'] = config['stage1_cleanspecnet']['batch_size']  # 64
        
        train_dataloader_stage1 = get_dataloader(config_stage1, split='train')
        val_dataloader_stage1 = get_dataloader(config_stage1, split='val')
        
        print(f"Stage 1 - CleanSpecNet data loaders created:")
        print(f"  Batch size: {config_stage1['batch_size']}")
        print(f"  Train batches: {len(train_dataloader_stage1)}")
        print(f"  Val batches: {len(val_dataloader_stage1)}")
    
    if args.stage in ['2', 'both']:
        # Stage 2: batch_size=32 (CleanUNet 2)
        config_stage2 = config.copy()
        config_stage2['batch_size'] = config['stage2_cleanunet2']['batch_size']  # 32
        
        train_dataloader_stage2 = get_dataloader(config_stage2, split='train')
        val_dataloader_stage2 = get_dataloader(config_stage2, split='val')
        
        print(f"Stage 2 - CleanUNet 2 data loaders created:")
        print(f"  Batch size: {config_stage2['batch_size']}")
        print(f"  Train batches: {len(train_dataloader_stage2)}")
        print(f"  Val batches: {len(val_dataloader_stage2)}")
    
    # Run training based on stage selection
    try:
        if args.stage == '1':
            print("\n🎯 Running STAGE 1 ONLY (CleanSpecNet training)")
            manager.train_stage1_cleanspecnet(train_dataloader_stage1, val_dataloader_stage1)
            
        elif args.stage == '2':
            print("\n🎯 Running STAGE 2 ONLY (CleanUNet 2 training)")
            if args.stage1_checkpoint:
                # Load stage 1 checkpoint
                print(f"Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
                checkpoint = torch.load(args.stage1_checkpoint, map_location=device)
                manager.model.clean_spec_net.load_state_dict(checkpoint['model_state_dict'])
                manager.cleanspecnet_trained = True
            else:
                print("ERROR: Stage 1 checkpoint required for Stage 2 training")
                print("Use --stage1_checkpoint to specify the path")
                return
                
            manager.train_stage2_cleanunet2(train_dataloader_stage2, val_dataloader_stage2)
            
        elif args.stage == 'both':
            print("\n🎯 Running COMPLETE TWO-STAGE PIPELINE")
            print("This will take a very long time (1M + 500K iterations)!")
            
            # Complete pipeline
            final_model = manager.train_full_pipeline(
                train_dataloader_stage1,  # Use stage 1 loader for both stages
                val_dataloader_stage1
            )
            print(f"\n🎉 Complete training finished!")
            print(f"Final model: {final_model}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()