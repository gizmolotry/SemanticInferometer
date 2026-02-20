"""
Attention Recorder - Simple storage for attention matrices
"""

from typing import Dict
import torch


class AttentionRecorder:
    """Records attention matrices from different months/batches."""
    
    def __init__(self):
        self.recordings = {}
    
    def record(self, key: str, attention_matrix: torch.Tensor):
        """Record an attention matrix with a key (e.g., month name)."""
        self.recordings[key] = attention_matrix.detach().cpu()
    
    def get(self, key: str):
        """Retrieve a recorded attention matrix."""
        return self.recordings.get(key)
    
    def keys(self):
        """Get all recorded keys."""
        return self.recordings.keys()
    
    def clear(self):
        """Clear all recordings."""
        self.recordings.clear()
