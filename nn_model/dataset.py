"""
Dataset utilities for neural-to-phoneme decoding
"""

import os
import h5py
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def discover_sessions(dataset_dir, split="train", exclude=None):
    """
    Automatically discover all valid sessions in dataset directory.
    
    Args:
        dataset_dir: Path to hdf5_data_final directory
        split: 'train', 'val', or 'test'
        exclude: Optional list of session names to exclude
        
    Returns:
        List of session names (sorted for reproducibility)
    """
    sessions = []
    exclude = exclude or []
    
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    for name in sorted(os.listdir(dataset_dir)):
        if name in exclude:
            continue
            
        p = os.path.join(dataset_dir, name)
        if not os.path.isdir(p):
            continue
            
        f = os.path.join(p, f"data_{split}.hdf5")
        if os.path.exists(f):
            sessions.append(name)
    
    return sessions


def build_trial_index(dataset_dir, sessions, split):
    """
    Build index mapping day_idx -> {trials, session_path}
    
    Args:
        dataset_dir: Path to hdf5_data_final directory
        sessions: List of session names
        split: 'train', 'val', or 'test'
        
    Returns:
        Dict: {day_idx: {"trials": [...], "session_path": "..."}}
    """
    index = {}
    
    for d, s in enumerate(sessions):
        fp = os.path.join(dataset_dir, s, f"data_{split}.hdf5")
        
        if not os.path.exists(fp):
            raise ValueError(f"Data file not found: {fp}")
        
        with h5py.File(fp, "r") as f:
            trials = [int(k.split("_")[1]) for k in f.keys() if k.startswith("trial_")]
            trials = sorted(trials)
        
        index[d] = {"trials": trials, "session_path": fp}
    
    return index


def infer_dims_and_classes(trial_index):
    """
    Infer neural_dim and n_classes from the first available trial.
    
    Args:
        trial_index: Output from build_trial_index()
        
    Returns:
        Tuple: (neural_dim, n_classes)
    """
    neural_dim = None
    max_label = 0
    
    for d in trial_index:
        fp = trial_index[d]["session_path"]
        
        with h5py.File(fp, "r") as f:
            for t in trial_index[d]["trials"]:
                g = f[f"trial_{t:04d}"]
                
                if neural_dim is None:
                    neural_dim = g["input_features"].shape[-1]
                
                lbl = g["seq_class_ids"][:]
                if len(lbl) > 0:
                    max_label = max(max_label, int(np.max(lbl)))
                
                break  # Only need one trial
        
        if neural_dim is not None:
            break
    
    n_classes = int(max_label) + 1  # Assume 0 is CTC blank
    
    return neural_dim, n_classes


class Neural2PhonemeBatchDataset(Dataset):
    """
    Dataset that returns entire batches (not single examples).
    This is necessary for handling variable-length sequences efficiently.
    
    Compatible with nejm-brain-to-text data format:
    - HDF5 files with trial_XXXX groups
    - Each trial has: input_features, seq_class_ids, n_time_steps, seq_len
    """

    def __init__(self,
                 dataset_dir,
                 split="train",
                 sessions=None,
                 n_batches=1000,
                 batch_size=64,
                 days_per_batch=1,
                 seed=-1,
                 feature_subset=None,
                 exclude_sessions=None):
        """
        Args:
            dataset_dir: Path to hdf5_data_final directory
            split: 'train', 'val', or 'test'
            sessions: List of session names (if None, auto-discover)
            n_batches: Number of training batches (ignored for val/test)
            batch_size: Trials per batch
            days_per_batch: Number of different days in each batch (train only)
            seed: Random seed for reproducibility
            feature_subset: Optional list of feature indices to use
            exclude_sessions: Optional list of sessions to exclude
        """
        self.split = split
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.feature_subset = feature_subset
        
        # Set random seed
        if seed != -1:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Discover or use provided sessions
        if sessions is None:
            self.sessions = discover_sessions(dataset_dir, split, exclude=exclude_sessions)
            if len(self.sessions) == 0:
                raise ValueError(f"No sessions found for split '{split}' in {dataset_dir}")
        else:
            self.sessions = sessions
        
        # Build trial index
        self.trial_index = build_trial_index(dataset_dir, self.sessions, split)
        self.n_days = len(self.sessions)
        self.n_trials = sum(len(self.trial_index[d]["trials"]) for d in self.trial_index)
        
        # Infer dimensions
        self.neural_dim, self.n_classes = infer_dims_and_classes(self.trial_index)
        
        # Create batch map
        if split == "train":
            self.days_per_batch = days_per_batch
            if self.days_per_batch > self.n_days:
                raise ValueError(f"days_per_batch ({days_per_batch}) > available days ({self.n_days})")
            self.n_batches = n_batches
            self.batch_map = self._make_train_batches()
        else:
            self.days_per_batch = 1
            self.batch_map = self._make_eval_batches()
            self.n_batches = len(self.batch_map)

    def __len__(self):
        return self.n_batches

    def _make_train_batches(self):
        """Create random batches for training."""
        m = {}
        day_indices = list(self.trial_index.keys())
        
        for bi in range(self.n_batches):
            # Randomly sample days
            days = np.random.choice(day_indices, size=self.days_per_batch, replace=False)
            per_day = math.ceil(self.batch_size / self.days_per_batch)
            
            batch = {}
            for d in days:
                trials = self.trial_index[d]["trials"]
                pick = np.random.choice(trials, size=per_day, replace=True)
                batch[int(d)] = pick.tolist()
            
            # Remove extra samples to match exact batch_size
            extra = (per_day * len(days)) - self.batch_size
            while extra > 0:
                d = int(np.random.choice(days))
                if len(batch[d]) > 0:
                    batch[d] = batch[d][:-1]
                    extra -= 1
            
            m[bi] = batch
        
        return m

    def _make_eval_batches(self):
        """Create sequential batches for validation/test."""
        m, bi = {}, 0
        
        for d in self.trial_index:
            trials = self.trial_index[d]["trials"]
            
            for start in range(0, len(trials), self.batch_size):
                batch_trials = trials[start:start + self.batch_size]
                m[bi] = {int(d): batch_trials}
                bi += 1
        
        return m

    def __getitem__(self, idx):
        """Load and return a complete batch."""
        batch = {
            "input_features": [],
            "seq_class_ids": [],
            "n_time_steps": [],
            "phone_seq_lens": [],
            "day_indicies": [],
            "block_nums": [],
            "trial_nums": [],
        }
        
        plan = self.batch_map[idx]
        
        for d in plan:
            with h5py.File(self.trial_index[d]["session_path"], "r") as f:
                for t in plan[d]:
                    try:
                        g = f[f"trial_{int(t):04d}"]
                        
                        # Load neural features
                        x = torch.from_numpy(g["input_features"][:]).float()
                        if self.feature_subset is not None:
                            x = x[:, self.feature_subset]
                        
                        # Load labels
                        y = torch.from_numpy(g["seq_class_ids"][:]).long()
                        
                        batch["input_features"].append(x)
                        batch["seq_class_ids"].append(y)
                        batch["n_time_steps"].append(int(g.attrs["n_time_steps"]))
                        batch["phone_seq_lens"].append(int(g.attrs["seq_len"]))
                        batch["day_indicies"].append(int(d))
                        batch["block_nums"].append(int(g.attrs["block_num"]))
                        batch["trial_nums"].append(int(g.attrs["trial_num"]))
                    
                    except Exception as e:
                        print(f"Warning: Error loading trial {t} from day {d}: {e}")
                        continue
        
        # Pad sequences to form cohesive batch
        batch["input_features"] = pad_sequence(batch["input_features"], batch_first=True, padding_value=0.0)
        batch["seq_class_ids"] = pad_sequence(batch["seq_class_ids"], batch_first=True, padding_value=0)
        batch["n_time_steps"] = torch.tensor(batch["n_time_steps"], dtype=torch.int32)
        batch["phone_seq_lens"] = torch.tensor(batch["phone_seq_lens"], dtype=torch.int32)
        batch["day_indicies"] = torch.tensor(batch["day_indicies"], dtype=torch.int64)
        batch["block_nums"] = torch.tensor(batch["block_nums"], dtype=torch.int32)
        batch["trial_nums"] = torch.tensor(batch["trial_nums"], dtype=torch.int32)
        
        return batch

