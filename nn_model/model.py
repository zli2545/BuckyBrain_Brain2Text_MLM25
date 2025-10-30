"""
GRU-based neural decoder model with day-specific input layers
"""

import torch
from torch import nn


class GRUDecoder(nn.Module):
    """
    GRU decoder with day-specific input layers for neural-to-phoneme decoding.
    
    Architecture:
    1. Day-specific linear layer (per-session calibration)
    2. Optional temporal patching (strided convolution-like)
    3. Multi-layer GRU
    4. Linear output layer (phoneme logits)
    
    Compatible with CTC loss (blank=0).
    """
    
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout=0.0,
                 input_dropout=0.0,
                 n_layers=5,
                 patch_size=0,
                 patch_stride=1):
        """
        Args:
            neural_dim: Number of input features (e.g., 512)
            n_units: Hidden units per GRU layer
            n_days: Number of recording sessions
            n_classes: Number of phoneme classes (including blank)
            rnn_dropout: Dropout between GRU layers
            input_dropout: Dropout after day-specific layer
            n_layers: Number of GRU layers
            patch_size: Temporal patch size (0=disabled)
            patch_stride: Stride for temporal patching
        """
        super().__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_days = n_days
        self.patch_size = patch_size
        self.patch_stride = max(1, patch_stride)
        self.input_dropout = input_dropout
        
        # Day-specific input layers (initialized as identity for stability)
        self.day_weights = nn.ParameterList([
            nn.Parameter(torch.eye(self.neural_dim)) 
            for _ in range(self.n_days)
        ])
        self.day_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.neural_dim)) 
            for _ in range(self.n_days)
        ])
        self.day_act = nn.Softsign()  # Shallower than tanh, more stable
        self.day_do = nn.Dropout(self.input_dropout)
        
        # Compute GRU input size
        in_size = self.neural_dim
        if self.patch_size > 0:
            in_size = self.neural_dim * self.patch_size
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=rnn_dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )
        
        # Initialize GRU weights
        for n, p in self.gru.named_parameters():
            if "weight_hh" in n:
                nn.init.orthogonal_(p)  # Recurrent weights
            if "weight_ih" in n:
                nn.init.xavier_uniform_(p)  # Input weights
        
        # Output layer
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)
        
        # Learnable initial hidden state
        self.h0 = nn.Parameter(torch.empty(1, 1, self.n_units))
        nn.init.xavier_uniform_(self.h0)

    def _apply_day_layers(self, x, day_idx):
        """Apply day-specific affine transformation."""
        # x: [B, T, D]
        # day_idx: [B]
        
        W = torch.stack([self.day_weights[i] for i in day_idx], dim=0)  # [B, D, D]
        b = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)  # [B, 1, D]
        
        x = torch.einsum("btd,bdk->btk", x, W) + b
        x = self.day_act(x)
        
        if self.input_dropout > 0:
            x = self.day_do(x)
        
        return x

    def _make_patches(self, x):
        """
        Create temporal patches by concatenating consecutive timesteps.
        Similar to strided convolution but concatenates instead of convolves.
        """
        if self.patch_size <= 0:
            return x
        
        # x: [B, T, D]
        B, T, D = x.shape
        
        x = x.unsqueeze(1).permute(0, 3, 1, 2)  # [B, D, 1, T]
        
        # Sliding window to extract patches
        u = x.unfold(3, self.patch_size, self.patch_stride)  # [B, D, 1, Np, P]
        
        u = u.squeeze(2).permute(0, 2, 3, 1)  # [B, Np, P, D]
        
        # Flatten patch and feature dimensions
        return u.reshape(B, u.size(1), -1)  # [B, Np, P*D]

    def forward(self, x, day_idx, states=None, return_state=False):
        """
        Forward pass.
        
        Args:
            x: Neural features [batch, time, features]
            day_idx: Day indices [batch]
            states: Optional initial hidden states
            return_state: Whether to return final hidden state
            
        Returns:
            logits: Phoneme logits [batch, time', n_classes]
            (optional) states: Final hidden states
        """
        # Apply day-specific transformation
        x = self._apply_day_layers(x, day_idx)
        
        # Apply temporal patching
        x = self._make_patches(x)
        
        # Initialize hidden states if needed
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()
        
        # Pass through GRU
        y, h = self.gru(x, states)
        
        # Compute phoneme logits
        logits = self.out(y)
        
        if return_state:
            return logits, h
        
        return logits
    
    def get_num_params(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_day_params(self):
        """Return number of day-specific parameters."""
        return sum(p.numel() for n, p in self.named_parameters() if 'day_' in n)

