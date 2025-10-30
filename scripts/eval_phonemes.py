#!/usr/bin/env python3
"""
Evaluation script for phoneme-level decoding
"""

import os
import sys
import numpy as np
import torch
from argparse import ArgumentParser
from torchaudio.functional import edit_distance

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from nn_model import Neural2PhonemeBatchDataset, GRUDecoder
from nn_model.utils import compute_adjusted_lengths, DataAugmentation


def main():
    parser = ArgumentParser(description="Evaluate neural-to-phoneme decoder")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint (e.g., outputs/best.pt)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to hdf5_data_final directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index (-1 for CPU)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    meta = ckpt['meta']
    
    print(f"Checkpoint info:")
    print(f"  Neural dim: {meta['neural_dim']}")
    print(f"  Classes: {meta['n_classes']}")
    print(f"  Model: {meta['n_layers']} layers, {meta['n_units']} units")
    print(f"  Patch: size={meta['patch_size']}, stride={meta['patch_stride']}")
    
    # Get sessions for this split
    if args.split == "train":
        sessions = meta.get('sessions_train', None)
    else:
        sessions = meta.get('sessions_val', None)
    
    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = Neural2PhonemeBatchDataset(
        dataset_dir=args.dataset_dir,
        split=args.split,
        sessions=sessions,
        n_batches=None,
        batch_size=args.batch_size
    )
    
    print(f"Dataset: {dataset.n_trials} trials across {dataset.n_days} days")
    
    # Create model
    model = GRUDecoder(
        neural_dim=meta['neural_dim'],
        n_units=meta['n_units'],
        n_days=dataset.n_days,  # Use dataset's n_days for flexibility
        n_classes=meta['n_classes'],
        n_layers=meta['n_layers'],
        rnn_dropout=0.0,  # No dropout during eval
        input_dropout=0.0,
        patch_size=meta['patch_size'],
        patch_stride=meta['patch_stride']
    )
    
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Create data augmentation (smoothing only for eval)
    augment = DataAugmentation(
        white_noise_std=0.0,
        constant_offset_std=0.0,
        smooth_data=True,
        smooth_kernel_std=2
    )
    
    # Evaluate
    print("\nEvaluating...")
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False)
    
    total_ed, total_len = 0, 0
    day_stats = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["input_features"].to(device)
            y = batch["seq_class_ids"].to(device)
            T = batch["n_time_steps"].to(device)
            L = batch["phone_seq_lens"].to(device)
            D = batch["day_indicies"].to(device)
            
            day_id = int(D[0].item())
            
            # Apply smoothing
            x, T = augment(x, T, mode='val', device=device)
            
            # Forward pass
            logits = model(x, D)
            
            # Compute adjusted lengths
            adj = compute_adjusted_lengths(T, meta['patch_size'], meta['patch_stride'])
            
            # Decode (greedy + CTC merge)
            batch_ed = 0
            for i in range(logits.shape[0]):
                li = int(adj[i].item())
                pred = torch.argmax(logits[i, :li, :], dim=-1)
                pred = torch.unique_consecutive(pred)
                pred = pred[pred != 0].cpu().numpy()
                
                tgt = y[i][:L[i]].cpu().numpy()
                
                ed = edit_distance(pred, tgt)
                batch_ed += ed
            
            # Accumulate stats
            if day_id not in day_stats:
                day_stats[day_id] = {'ed': 0, 'len': 0}
            
            day_stats[day_id]['ed'] += batch_ed
            day_stats[day_id]['len'] += torch.sum(L).item()
            
            total_ed += batch_ed
            total_len += torch.sum(L).item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    # Compute PER
    overall_per = total_ed / max(1, total_len)
    
    print("\n" + "="*80)
    print(f"Results on {args.split} split:")
    print("="*80)
    print(f"Overall PER: {overall_per:.4f} ({total_ed} errors / {total_len} phonemes)")
    print("\nPer-day PER:")
    
    for day_id in sorted(day_stats.keys()):
        stats = day_stats[day_id]
        per = stats['ed'] / max(1, stats['len'])
        session_name = dataset.sessions[day_id] if day_id < len(dataset.sessions) else f"day_{day_id}"
        print(f"  {session_name:30s} PER: {per:.4f} ({stats['ed']:4d} / {stats['len']:5d})")
    
    print("="*80)


if __name__ == "__main__":
    main()

