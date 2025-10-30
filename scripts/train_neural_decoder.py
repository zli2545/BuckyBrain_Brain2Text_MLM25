#!/usr/bin/env python3
"""
Training script for BuckyBrain neural decoder
"""

import os
import sys
from argparse import ArgumentParser
from omegaconf import OmegaConf

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from nn_model import Trainer


def main():
    parser = ArgumentParser(description="Train neural-to-phoneme decoder")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/neural_decoder.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Override dataset directory from config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="Override GPU index from config"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Quick debug mode (small batch count)"
    )
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    cfg = OmegaConf.load(args.config)
    
    # Apply command-line overrides
    if args.dataset_dir:
        cfg.dataset.dataset_dir = args.dataset_dir
    if args.output_dir:
        cfg.system.output_dir = args.output_dir
    if args.gpu is not None:
        cfg.system.gpu = args.gpu
    
    # Debug mode: reduce batch count
    if args.debug:
        print("DEBUG MODE: Using reduced batch count")
        cfg.training.n_train_batches = 500
        cfg.training.val_every = 100
        if cfg.system.output_dir == "./outputs":
            cfg.system.output_dir = "./outputs_debug"
    
    # Print configuration
    print("="*80)
    print("Configuration:")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)
    
    # Create trainer
    trainer = Trainer(
        # Dataset
        dataset_dir=cfg.dataset.dataset_dir,
        sessions_train=cfg.dataset.sessions_train,
        sessions_val=cfg.dataset.sessions_val,
        exclude_sessions=cfg.dataset.exclude_sessions,
        auto_discover=cfg.dataset.auto_discover,
        # Model
        n_units=cfg.model.n_units,
        n_layers=cfg.model.n_layers,
        rnn_dropout=cfg.model.rnn_dropout,
        input_dropout=cfg.model.input_dropout,
        patch_size=cfg.model.patch_size,
        patch_stride=cfg.model.patch_stride,
        # Training
        batch_size=cfg.training.batch_size,
        days_per_batch=cfg.training.days_per_batch,
        n_train_batches=cfg.training.n_train_batches,
        val_every=cfg.training.val_every,
        # Optimizer
        lr=cfg.optimizer.lr,
        lr_min=cfg.optimizer.lr_min,
        lr_day=cfg.optimizer.lr_day,
        weight_decay=cfg.optimizer.weight_decay,
        weight_decay_day=cfg.optimizer.weight_decay_day,
        grad_clip=cfg.optimizer.grad_clip,
        # Augmentation
        white_noise_std=cfg.augmentation.white_noise_std,
        constant_offset_std=cfg.augmentation.constant_offset_std,
        random_walk_std=cfg.augmentation.random_walk_std,
        static_gain_std=cfg.augmentation.static_gain_std,
        random_cut=cfg.augmentation.random_cut,
        smooth_data=cfg.augmentation.smooth_data,
        smooth_kernel_std=cfg.augmentation.smooth_kernel_std,
        # System
        output_dir=cfg.system.output_dir,
        amp=cfg.system.amp,
        use_compile=cfg.system.use_compile,
        seed=cfg.system.seed,
        gpu=cfg.system.gpu,
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()

