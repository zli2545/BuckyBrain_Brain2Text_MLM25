"""
Trainer for neural-to-phoneme decoder
"""

import os
import sys
import time
import math
import logging
import pathlib
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchaudio.functional import edit_distance
from omegaconf import OmegaConf

from .dataset import Neural2PhonemeBatchDataset, discover_sessions
from .model import GRUDecoder
from .utils import DataAugmentation, compute_adjusted_lengths


class Trainer:
    """
    Trainer for GRU-based neural decoder with comprehensive logging and checkpointing.
    """
    
    def __init__(self,
                 dataset_dir,
                 output_dir="./outputs",
                 # Session configuration
                 sessions_train=None,
                 sessions_val=None,
                 exclude_sessions=None,
                 auto_discover=False,
                 # Model configuration
                 n_units=768,
                 n_layers=5,
                 rnn_dropout=0.4,
                 input_dropout=0.2,
                 patch_size=14,
                 patch_stride=4,
                 # Training configuration
                 batch_size=64,
                 days_per_batch=1,
                 n_train_batches=10000,
                 val_every=500,
                 # Optimizer configuration
                 lr=0.005,
                 lr_min=0.0001,
                 lr_day=0.005,
                 weight_decay=0.001,
                 weight_decay_day=0.0,
                 grad_clip=10.0,
                 # Data augmentation
                 white_noise_std=1.0,
                 constant_offset_std=0.2,
                 random_walk_std=0.0,
                 static_gain_std=0.0,
                 random_cut=3,
                 smooth_data=True,
                 smooth_kernel_std=2,
                 # System configuration
                 amp=True,
                 use_compile=False,
                 seed=42,
                 gpu=0):
        """
        Args:
            dataset_dir: Path to hdf5_data_final directory
            output_dir: Directory for logs and checkpoints
            sessions_train: List of training session names (None=auto-discover)
            sessions_val: List of validation session names (None=auto-discover)
            exclude_sessions: Sessions to exclude from auto-discovery
            auto_discover: Use auto-discovery even if sessions provided
            n_units: GRU hidden units per layer
            n_layers: Number of GRU layers
            rnn_dropout: Dropout between GRU layers
            input_dropout: Dropout after day-specific layer
            patch_size: Temporal patch size (0=disabled)
            patch_stride: Stride for patching
            batch_size: Samples per batch
            days_per_batch: Different days per training batch
            n_train_batches: Total training batches
            val_every: Validate every N batches
            lr: Base learning rate
            lr_min: Minimum learning rate (for scheduler)
            lr_day: Learning rate for day-specific layers
            weight_decay: L2 regularization for main parameters
            weight_decay_day: L2 regularization for day layers
            grad_clip: Gradient clipping threshold
            white_noise_std: Std of white noise augmentation
            constant_offset_std: Std of offset augmentation
            random_walk_std: Std of random walk augmentation
            static_gain_std: Std of gain augmentation
            random_cut: Max timesteps to cut from start
            smooth_data: Apply Gaussian smoothing
            smooth_kernel_std: Std for smoothing kernel
            amp: Use automatic mixed precision
            use_compile: Use torch.compile (PyTorch 2.0+)
            seed: Random seed
            gpu: GPU index (-1 for CPU)
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Device configuration
        self.device = self._setup_device(gpu)
        
        # Auto-detect device type for AMP (CPU doesn't support bfloat16)
        if self.device_type == "cpu":
            # CPU: disable AMP completely (CPU autocast only supports bfloat16/float16, not float32)
            if amp:
                self.logger.warning("AMP with bfloat16 not supported on CPU. Using float32 instead.")
                self.amp = False  # Disable AMP on CPU
                self.amp_dtype = torch.float32
            else:
                self.amp_dtype = torch.float32
        else:
            # CUDA: use bfloat16
            self.amp_dtype = torch.bfloat16 if amp else torch.float32
        
        # Set random seed
        if seed != -1:
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.logger.info(f"Random seed: {seed}")
        
        # Session discovery
        if auto_discover or sessions_train is None:
            self.sessions_train = discover_sessions(dataset_dir, "train", exclude=exclude_sessions)
            self.logger.info(f"Auto-discovered {len(self.sessions_train)} training sessions")
        else:
            self.sessions_train = sessions_train
        
        if auto_discover or sessions_val is None:
            self.sessions_val = discover_sessions(dataset_dir, "val", exclude=exclude_sessions)
            self.logger.info(f"Auto-discovered {len(self.sessions_val)} validation sessions")
        else:
            self.sessions_val = sessions_val
        
        self.logger.info(f"Training sessions: {self.sessions_train[:3]}..." if len(self.sessions_train) > 3 else f"Training sessions: {self.sessions_train}")
        self.logger.info(f"Validation sessions: {self.sessions_val[:3]}..." if len(self.sessions_val) > 3 else f"Validation sessions: {self.sessions_val}")
        
        # Create datasets
        self.train_ds = Neural2PhonemeBatchDataset(
            dataset_dir=dataset_dir,
            split="train",
            sessions=self.sessions_train,
            n_batches=n_train_batches,
            batch_size=batch_size,
            days_per_batch=days_per_batch,
            seed=seed
        )
        
        self.val_ds = Neural2PhonemeBatchDataset(
            dataset_dir=dataset_dir,
            split="val",
            sessions=self.sessions_val,
            n_batches=None,
            batch_size=batch_size,
            days_per_batch=1,
            seed=seed
        )
        
        self.logger.info(f"Training: {self.train_ds.n_trials} trials across {self.train_ds.n_days} days")
        self.logger.info(f"Validation: {self.val_ds.n_trials} trials across {self.val_ds.n_days} days")
        self.logger.info(f"Neural dim: {self.train_ds.neural_dim}, Classes: {self.train_ds.n_classes}")
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_ds, 
            batch_size=None, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_ds, 
            batch_size=None, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
        
        # Create model
        self.model = GRUDecoder(
            neural_dim=self.train_ds.neural_dim,
            n_units=n_units,
            n_days=len(self.sessions_train),
            n_classes=self.train_ds.n_classes,
            rnn_dropout=rnn_dropout,
            input_dropout=input_dropout,
            n_layers=n_layers,
            patch_size=patch_size,
            patch_stride=patch_stride
        ).to(self.device)
        
        total_params = self.model.get_num_params()
        day_params = self.model.get_num_day_params()
        self.logger.info(f"Model parameters: {total_params:,} total, {day_params:,} day-specific ({100*day_params/total_params:.1f}%)")
        
        # Optional torch.compile
        if use_compile:
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Using torch.compile")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}, using eager mode")
        
        # Create optimizer with parameter groups
        self.optim = self._create_optimizer(lr, lr_day, weight_decay, weight_decay_day)
        
        # CTC loss
        self.ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)
        
        # Data augmentation
        self.augment = DataAugmentation(
            white_noise_std=white_noise_std,
            constant_offset_std=constant_offset_std,
            random_walk_std=random_walk_std,
            static_gain_std=static_gain_std,
            random_cut=random_cut,
            smooth_data=smooth_data,
            smooth_kernel_std=smooth_kernel_std
        )
        
        # Training configuration
        self.n_train_batches = n_train_batches
        self.val_every = val_every
        self.grad_clip = grad_clip
        self.amp = amp
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
        # Tracking
        self.best_per = math.inf
        self.best_loss = math.inf

    def _setup_logger(self):
        """Setup logging to file and console."""
        logger = logging.getLogger(__name__)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.output_dir, 'training.log'))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        return logger

    def _setup_device(self, gpu):
        """Setup compute device."""
        if torch.cuda.is_available() and gpu >= 0:
            if gpu >= torch.cuda.device_count():
                self.logger.warning(f"GPU {gpu} not available, using GPU 0")
                gpu = 0
            device = torch.device(f"cuda:{gpu}")
            self.device_type = "cuda"
            self.logger.info(f"Using device: {device} ({torch.cuda.get_device_name(gpu)})")
        else:
            device = torch.device("cpu")
            self.device_type = "cpu"
            self.logger.info("Using device: CPU")
        
        return device

    def _create_optimizer(self, lr, lr_day, weight_decay, weight_decay_day):
        """Create optimizer with parameter groups."""
        bias_params = []
        day_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name:
                bias_params.append(param)
            elif 'day_' in name:
                day_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': bias_params, 'weight_decay': 0.0, 'name': 'bias'},
            {'params': day_params, 'lr': lr_day, 'weight_decay': weight_decay_day, 'name': 'day'},
            {'params': other_params, 'weight_decay': weight_decay, 'name': 'other'}
        ]
        
        optimizer = torch.optim.AdamW(param_groups, lr=lr, fused=torch.cuda.is_available())
        
        self.logger.info(f"Optimizer groups: bias={len(bias_params)}, day={len(day_params)}, other={len(other_params)}")
        
        return optimizer

    @torch.no_grad()
    def validate(self):
        """Run validation and compute per-day and aggregate PER."""
        self.model.eval()
        
        total_ed, total_len = 0.0, 0.0
        total_loss = []
        day_stats = {}
        
        for batch in self.val_loader:
            x = batch["input_features"].to(self.device)
            y = batch["seq_class_ids"].to(self.device)
            T = batch["n_time_steps"].to(self.device)
            L = batch["phone_seq_lens"].to(self.device)
            D = batch["day_indicies"].to(self.device)
            
            day_id = int(D[0].item())
            
            with torch.autocast(device_type=self.device_type, enabled=self.amp, dtype=self.amp_dtype):
                # Apply augmentation (validation mode = smoothing only)
                x, T = self.augment(x, T, mode='val', device=self.device)
                
                # Forward pass
                logits = self.model(x, D)
                
                # Compute adjusted lengths after patching
                adj = compute_adjusted_lengths(T, self.patch_size, self.patch_stride)
                
                # CTC loss
                loss = self.ctc(
                    torch.permute(logits.log_softmax(2), (1, 0, 2)),
                    y,
                    adj,
                    L
                ).mean()
            
            total_loss.append(loss.item())
            
            # Decode predictions (greedy + CTC merge)
            batch_ed = 0
            for i in range(logits.shape[0]):
                li = int(adj[i].item())
                pred = torch.argmax(logits[i, :li, :], dim=-1)
                pred = torch.unique_consecutive(pred)
                pred = pred[pred != 0].cpu().numpy()
                
                tgt = y[i][:L[i]].cpu().numpy()
                
                ed = edit_distance(pred, tgt)
                batch_ed += ed
            
            # Accumulate per-day statistics
            if day_id not in day_stats:
                day_stats[day_id] = {'ed': 0, 'len': 0}
            
            day_stats[day_id]['ed'] += batch_ed
            day_stats[day_id]['len'] += torch.sum(L).item()
            
            total_ed += batch_ed
            total_len += torch.sum(L).item()
        
        # Compute metrics
        per = total_ed / max(1.0, total_len)
        avg_loss = float(np.mean(total_loss)) if total_loss else 0.0
        
        # Per-day PER
        day_per = {}
        for d, stats in day_stats.items():
            if stats['len'] > 0:
                day_per[d] = stats['ed'] / stats['len']
        
        return {
            'per': per,
            'loss': avg_loss,
            'day_per': day_per,
            'day_stats': day_stats
        }

    def train(self):
        """Main training loop."""
        self.logger.info("="*80)
        self.logger.info("Starting training")
        self.logger.info("="*80)
        
        scaler_ctx = torch.autocast(device_type=self.device_type, enabled=self.amp, dtype=self.amp_dtype)
        self.model.train()
        
        t0 = time.time()
        
        for step, batch in enumerate(self.train_loader, start=1):
            x = batch["input_features"].to(self.device)
            y = batch["seq_class_ids"].to(self.device)
            T = batch["n_time_steps"].to(self.device)
            L = batch["phone_seq_lens"].to(self.device)
            D = batch["day_indicies"].to(self.device)
            
            self.optim.zero_grad(set_to_none=True)
            
            with scaler_ctx:
                # Apply augmentation (training mode)
                x, T = self.augment(x, T, mode='train', device=self.device)
                
                # Forward pass
                logits = self.model(x, D)
                
                # Compute adjusted lengths
                adj = compute_adjusted_lengths(T, self.patch_size, self.patch_stride)
                
                # CTC loss
                loss = self.ctc(
                    torch.permute(logits.log_softmax(2), (1, 0, 2)),
                    y,
                    adj,
                    L
                ).mean()
            
            loss.backward()
            
            # Gradient clipping
            grad_norm = 0.0
            if self.grad_clip and self.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.grad_clip,
                    error_if_nonfinite=True
                )
            
            self.optim.step()
            
            # Logging
            if step % 100 == 0:
                self.logger.info(
                    f"Train [{step}/{self.n_train_batches}] "
                    f"loss: {loss.item():.4f} "
                    f"grad_norm: {grad_norm:.2f}"
                )
            
            # Validation
            if (step % self.val_every == 0) or (step == self.n_train_batches):
                val_metrics = self.validate()
                
                self.logger.info(
                    f"Val [{step}] "
                    f"PER: {val_metrics['per']:.4f} "
                    f"loss: {val_metrics['loss']:.4f}"
                )
                
                # Log per-day PER
                for d, per in val_metrics['day_per'].items():
                    session_name = self.sessions_val[d] if d < len(self.sessions_val) else f"day_{d}"
                    self.logger.info(f"  {session_name}: PER={per:.4f}")
                
                # Save best checkpoint
                new_best = False
                if val_metrics['per'] < self.best_per:
                    self.logger.info(f"New best PER: {self.best_per:.4f} -> {val_metrics['per']:.4f}")
                    self.best_per = val_metrics['per']
                    self.best_loss = val_metrics['loss']
                    new_best = True
                elif val_metrics['per'] == self.best_per and val_metrics['loss'] < self.best_loss:
                    self.logger.info(f"New best loss: {self.best_loss:.4f} -> {val_metrics['loss']:.4f}")
                    self.best_loss = val_metrics['loss']
                    new_best = True
                
                if new_best:
                    self._save_checkpoint('best.pt', val_metrics)
                
                self.model.train()
            
            if step >= self.n_train_batches:
                break
        
        elapsed = (time.time() - t0) / 60
        self.logger.info("="*80)
        self.logger.info(f"Training complete in {elapsed:.1f} minutes")
        self.logger.info(f"Best PER: {self.best_per:.4f}")
        self.logger.info("="*80)

    def _save_checkpoint(self, filename, metrics):
        """Save model checkpoint."""
        ckpt_path = os.path.join(self.output_dir, filename)
        
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'metrics': metrics,
            'meta': {
                'sessions_train': self.sessions_train,
                'sessions_val': self.sessions_val,
                'neural_dim': self.train_ds.neural_dim,
                'n_classes': self.train_ds.n_classes,
                'n_units': self.model.n_units,
                'n_layers': self.model.n_layers,
                'patch_size': self.model.patch_size,
                'patch_stride': self.model.patch_stride,
            }
        }
        
        torch.save(checkpoint, ckpt_path)
        self.logger.info(f"Saved checkpoint: {ckpt_path}")

