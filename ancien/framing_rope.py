"""
Fixed Framing RoPE - Handles Odd Dimensions

Changes:
- Automatically pads odd dimensions to even for rotation
- Strips padding after rotation
- Preserves original dimensionality
"""

import torch
import torch.nn as nn
import math


class RotationLayer(nn.Module):
    """
    Applies 2D rotation to pairs of features.
    Now handles odd dimensions by padding.
    """
    
    def __init__(self, dim: int, angle: float):
        super().__init__()
        
        self.original_dim = dim
        
        # Pad to even if needed
        self.needs_padding = (dim % 2 != 0)
        if self.needs_padding:
            self.padded_dim = dim + 1
            print(f"    [RoPE] Padding {dim}D → {self.padded_dim}D for rotation")
        else:
            self.padded_dim = dim
        
        # Create rotation matrices for pairs
        self.angle = angle
        n_pairs = self.padded_dim // 2
        
        # Stack rotation matrices for all pairs
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)
        
        # Rotation matrix for each pair
        rotation = torch.tensor([
            [cos_val, -sin_val],
            [sin_val, cos_val]
        ], dtype=torch.float32)
        
        # Expand to all pairs: [n_pairs, 2, 2]
        self.register_buffer('rotation', rotation.unsqueeze(0).expand(n_pairs, 2, 2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to feature pairs.
        
        Args:
            x: [batch, original_dim]
        
        Returns:
            [batch, original_dim] (same shape)
        """
        batch_size = x.shape[0]
        
        # Pad if needed
        if self.needs_padding:
            # Pad with zeros: [batch, dim] → [batch, dim+1]
            x_padded = torch.cat([x, torch.zeros(batch_size, 1, device=x.device)], dim=-1)
        else:
            x_padded = x
        
        # Reshape to pairs: [batch, n_pairs, 2]
        n_pairs = self.padded_dim // 2
        x_pairs = x_padded.reshape(batch_size, n_pairs, 2)
        
        # Apply rotation: [batch, n_pairs, 2] @ [n_pairs, 2, 2] → [batch, n_pairs, 2]
        # Use einsum for batched matrix multiply
        rotated = torch.einsum('bpi,pij->bpj', x_pairs, self.rotation)
        
        # Flatten back: [batch, n_pairs, 2] → [batch, padded_dim]
        # Use reshape instead of view for non-contiguous tensors
        x_rotated = rotated.reshape(batch_size, self.padded_dim)
        
        # Strip padding if we added it
        if self.needs_padding:
            x_rotated = x_rotated[:, :self.original_dim]
        
        return x_rotated


class FramingRoPE(nn.Module):
    """
    Framing-aware Rotary Position Encoding.
    
    Now handles odd dimensions by padding internally.
    """
    
    def __init__(
        self,
        feature_dim: int,
        n_framings: int,
        max_angle: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_framings = n_framings
        
        # Check if divisible by n_framings
        if feature_dim % n_framings != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by n_framings ({n_framings})"
            )
        
        self.dim_per_framing = feature_dim // n_framings
        
        print(f"\n### Framing RoPE ###")
        print(f"  Total dim: {feature_dim}")
        print(f"  Framings: {n_framings}")
        print(f"  Dim per framing: {self.dim_per_framing}")
        if self.dim_per_framing % 2 != 0:
            print(f"  ⚠ Odd dimension ({self.dim_per_framing}) - will pad internally")
        print(f"  Max angle: {max_angle} rad")
        
        # Create rotation layers for each framing
        # Angle increases linearly with framing index
        self.rotations = nn.ModuleList([
            RotationLayer(
                dim=self.dim_per_framing,
                angle=max_angle * (i / n_framings)
            )
            for i in range(n_framings)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply framing-specific rotations.
        
        Args:
            x: [batch, feature_dim]
        
        Returns:
            [batch, feature_dim]
        """
        batch_size = x.shape[0]
        
        # Split into framings: [batch, n_framings, dim_per_framing]
        x_framings = x.reshape(batch_size, self.n_framings, self.dim_per_framing)
        
        # Apply rotation to each framing
        rotated_framings = []
        for i in range(self.n_framings):
            # Extract this framing: [batch, dim_per_framing]
            x_framing = x_framings[:, i, :]
            
            # Rotate
            x_rotated = self.rotations[i](x_framing)
            
            rotated_framings.append(x_rotated)
        
        # Stack and flatten: [batch, n_framings, dim_per_framing] → [batch, feature_dim]
        x_rotated = torch.stack(rotated_framings, dim=1)
        x_rotated = x_rotated.reshape(batch_size, self.feature_dim)
        
        return x_rotated


if __name__ == "__main__":
    print("="*70)
    print("Testing Fixed Framing RoPE with Odd Dimensions")
    print("="*70)
    
    # Test case 1: Even dimensions (original case)
    print("\n[Test 1] Even dimensions (8 framings × 4 dims = 32D):")
    rope_even = FramingRoPE(feature_dim=32, n_framings=8, max_angle=0.1)
    x_even = torch.randn(10, 32)
    y_even = rope_even(x_even)
    print(f"  Input:  {x_even.shape}")
    print(f"  Output: {y_even.shape}")
    print(f"  ✓ Works with even dims")
    
    # Test case 2: Odd dimensions (logits case)
    print("\n[Test 2] Odd dimensions (8 framings × 3 logits = 24D):")
    rope_odd = FramingRoPE(feature_dim=24, n_framings=8, max_angle=0.1)
    x_odd = torch.randn(10, 24)
    y_odd = rope_odd(x_odd)
    print(f"  Input:  {x_odd.shape}")
    print(f"  Output: {y_odd.shape}")
    print(f"  ✓ Works with odd dims!")
    
    # Verify output shape matches input
    assert y_odd.shape == x_odd.shape, "Output shape mismatch!"
    print("\n✓✓✓ All tests passed!")
    
    print("\n" + "="*70)