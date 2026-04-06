"""
Core neural network layers for the attention-based MCS selection model.
Includes Multi-Head Self-Attention and Positional Encoding adapted for OFDM time-frequency data.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for temporal sequences of OFDM symbols.
    Unlike sinusoidal encodings, these are learned during training.
    Adapted from Channelformer (Paper 8) for OFDM numerology.
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, d_model]
        Returns:
            x + positional encoding (truncated/padded to seq_len)
        """
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism (standard Transformer).
    Uses scaled dot-product attention.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Query, Key, Value projections (linear layers)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for dot product
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: [batch, seq_len, d_model]
            attn_mask: [batch, n_heads, seq_len, seq_len] or None
        Returns:
            output: [batch, seq_len, d_model]
            attn_weights: [batch, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)

        # Project and reshape: [batch, seq_len, d_model] -> [batch, n_heads, seq_len, d_k]
        q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, n_heads, seq_len, seq_len]

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, d_k]

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)

        return output, attn_weights


class TransformerEncoderLayerPreLN(nn.Module):
    """
    Transformer encoder layer with Pre-LayerNorm (pre-LN) architecture.
    Pre-LN provides more stable training than post-LN (original Transformer).
    Structure:
        x -> LayerNorm -> Self-Attention -> Residual
        -> LayerNorm -> Feed-Forward -> Residual
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Self-attention
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU often outperforms ReLU in Transformers
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            attn_mask: optional attention mask
            return_attn: if True, returns attention weights (for visualization)
        Returns:
            x: [batch, seq_len, d_model]
            (optional) attn_weights: [batch, n_heads, seq_len, seq_len]
        """
        # Pre-LN: Attention
        residual = x
        x = self.norm1(x)
        attn_output, attn_weights = self.self_attn(x, x, x, attn_mask)
        x = residual + self.dropout(attn_output)

        # Pre-LN: Feed-Forward
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + self.dropout(ff_output)

        if return_attn:
            return x, attn_weights
        return x


class TemporalAttentionTransformer(nn.Module):
    """
    Full Transformer encoder for MCS selection.
    Applies multi-head self-attention over temporal dimension (past OFDM symbols).
    """
    def __init__(
        self,
        input_dim: int,  # n_subcarriers * 2 (real/imag) or n_subcarriers if using magnitude
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_classes: int = 16,
        max_pos_encoding: int = 100
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Input projection: flatten subcarrier dimension into d_model
        # Input shape: [batch, seq_len, n_subcarriers, 2] -> [batch, seq_len, n_subcarriers*2]
        # Then project to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional encoding
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len=max_pos_encoding)

        # Transformer encoder layers (pre-LN)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerPreLN(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input channel tensor of shape [batch, seq_len, n_subcarriers, 2]
               (real and imaginary parts)
            attn_mask: Optional attention mask [batch, seq_len, seq_len] or None
            return_attn: If True, returns attention weights from all layers
        Returns:
            logits: [batch, n_classes] (pre-softmax scores)
            (optional) attn_weights_list: list of attention weight tensors
        """
        batch_size = x.size(0)

        # Flatten subcarrier and complex dimensions: [batch, seq_len, n_subcarriers*2]
        x = x.view(batch_size, self.seq_len, -1)

        # Project to model dimension
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer encoder layers
        attn_weights_list = [] if return_attn else None
        for layer in self.encoder_layers:
            if return_attn:
                x, attn_weights = layer(x, attn_mask, return_attn=True)
                attn_weights_list.append(attn_weights)
            else:
                x = layer(x, attn_mask, return_attn=False)

        # Final normalization
        x = self.final_norm(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [batch, d_model]

        # Classification
        logits = self.classifier(x)  # [batch, n_classes]

        if return_attn:
            return logits, attn_weights_list
        return logits


def create_attention_mask(
    seq_len: int,
    batch_size: int,
    device: torch.device,
    mask_type: str = "causal"
) -> torch.Tensor:
    """
    Create attention mask.
    Args:
        seq_len: Length of sequence
        batch_size: Batch size
        device: torch.device
        mask_type: 'causal' (no looking ahead) or None (full attention)
    Returns:
        mask: [batch, 1, seq_len, seq_len] (additive mask: 0 for allowed, -1e9 for masked)
    """
    if mask_type == "causal":
        # Causal mask: each position can only attend to previous positions
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, 1, seq_len, seq_len)
        # Convert to additive mask
        mask = (mask == 0) * (-1e9) + (mask == 1) * 0.0
        return mask
    else:
        # No mask (full attention)
        return None