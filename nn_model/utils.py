"""
Utility functions for data augmentation and transforms
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d


def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100, padding='same'):
    """
    Apply 1D Gaussian smoothing along time axis.
    
    Args:
        inputs: Tensor [B, T, N] - batch, time, features
        device: torch device
        smooth_kernel_std: Standard deviation of Gaussian kernel
        smooth_kernel_size: Size of kernel (will be truncated)
        padding: 'same' or 'valid'
        
    Returns:
        Smoothed tensor [B, T, N]
    """
    # Create Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gauss_kernel = gaussian_filter1d(inp, smooth_kernel_std)
    
    # Truncate kernel (keep only values > 0.01)
    valid_idx = np.argwhere(gauss_kernel > 0.01)
    gauss_kernel = gauss_kernel[valid_idx]
    gauss_kernel = np.squeeze(gauss_kernel / np.sum(gauss_kernel))
    
    # Convert to torch tensor
    gauss_kernel = torch.tensor(gauss_kernel, dtype=torch.float32, device=device)
    gauss_kernel = gauss_kernel.view(1, 1, -1)  # [1, 1, K]
    
    # Prepare for depthwise convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gauss_kernel = gauss_kernel.repeat(C, 1, 1)  # [C, 1, K]
    
    # Apply convolution
    smoothed = F.conv1d(inputs, gauss_kernel, padding=padding, groups=C)
    
    return smoothed.permute(0, 2, 1)  # [B, T, C]


class DataAugmentation:
    """
    Data augmentation for neural signals.
    Includes noise injection, gain variation, and smoothing.
    """
    
    def __init__(self,
                 white_noise_std=1.0,
                 constant_offset_std=0.2,
                 random_walk_std=0.0,
                 static_gain_std=0.0,
                 random_cut=3,
                 smooth_data=True,
                 smooth_kernel_std=2,
                 smooth_kernel_size=100):
        """
        Args:
            white_noise_std: Std of additive Gaussian noise
            constant_offset_std: Std of constant offset per trial
            random_walk_std: Std of random walk noise
            static_gain_std: Std of static gain variation
            random_cut: Max timesteps to randomly cut from start
            smooth_data: Whether to apply Gaussian smoothing
            smooth_kernel_std: Std for Gaussian smoothing
            smooth_kernel_size: Kernel size for smoothing
        """
        self.white_noise_std = white_noise_std
        self.constant_offset_std = constant_offset_std
        self.random_walk_std = random_walk_std
        self.static_gain_std = static_gain_std
        self.random_cut = random_cut
        self.smooth_data = smooth_data
        self.smooth_kernel_std = smooth_kernel_std
        self.smooth_kernel_size = smooth_kernel_size
    
    def __call__(self, features, n_time_steps, mode='train', device='cuda'):
        """
        Apply augmentations to neural features.
        
        Args:
            features: Tensor [B, T, C]
            n_time_steps: Tensor [B] - actual timesteps per trial
            mode: 'train' or 'val' (val only applies smoothing)
            device: torch device
            
        Returns:
            Augmented features, adjusted n_time_steps
        """
        B, T, C = features.shape
        
        # Training augmentations
        if mode == 'train':
            # Static gain noise (per-trial scale variation)
            if self.static_gain_std > 0:
                warp_mat = torch.eye(C, device=device).unsqueeze(0).repeat(B, 1, 1)
                warp_mat += torch.randn_like(warp_mat) * self.static_gain_std
                features = torch.matmul(features, warp_mat)
            
            # White noise
            if self.white_noise_std > 0:
                features += torch.randn_like(features) * self.white_noise_std
            
            # Constant offset (per-trial DC shift)
            if self.constant_offset_std > 0:
                offset = torch.randn(B, 1, C, device=device) * self.constant_offset_std
                features += offset
            
            # Random walk noise
            if self.random_walk_std > 0:
                walk = torch.cumsum(
                    torch.randn_like(features) * self.random_walk_std, 
                    dim=1
                )
                features += walk
            
            # Random temporal cutoff
            if self.random_cut > 0:
                cut = np.random.randint(0, self.random_cut)
                if cut > 0:
                    features = features[:, cut:, :]
                    n_time_steps = n_time_steps - cut
        
        # Gaussian smoothing (both train and val)
        if self.smooth_data:
            features = gauss_smooth(
                features,
                device=device,
                smooth_kernel_std=self.smooth_kernel_std,
                smooth_kernel_size=self.smooth_kernel_size
            )
        
        return features, n_time_steps


def compute_adjusted_lengths(n_time_steps, patch_size, patch_stride):
    """
    Compute sequence lengths after temporal patching.
    
    Formula: (T - P) / S + 1
    
    Args:
        n_time_steps: Original sequence lengths [B]
        patch_size: Patch size
        patch_stride: Patch stride
        
    Returns:
        Adjusted lengths [B]
    """
    if patch_size <= 0:
        return n_time_steps.to(torch.int32)
    
    T = n_time_steps.to(torch.int32)
    adjusted = ((T - patch_size) // patch_stride + 1).clamp(min=1)
    
    return adjusted

