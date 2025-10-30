"""
BuckyBrain Neural Decoder Package
Neural activity to phoneme decoding with GRU+CTC
"""

from .dataset import Neural2PhonemeBatchDataset, discover_sessions, build_trial_index
from .model import GRUDecoder
from .trainer import Trainer

__version__ = "0.1.0"
__all__ = [
    "Neural2PhonemeBatchDataset",
    "discover_sessions", 
    "build_trial_index",
    "GRUDecoder",
    "Trainer"
]

