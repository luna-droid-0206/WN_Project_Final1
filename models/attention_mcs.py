"""
Attention-based MCS selection model.
Primary contribution: Multi-head self-attention over temporal dimension.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .layers import TemporalAttentionTransformer, create_attention_mask
from training.config import AttentionModelConfig, DataConfig


class AttentionMCSModel(nn.Module):
    """
    Wrapper class for the attention-based MCS selection model.
    Incorporates configuration from config.py and provides convenient interface.
    """
    def __init__(
        self,
        config: Optional[AttentionModelConfig] = None,
        data_config: Optional[DataConfig] = None,
        **kwargs
    ):
        super().__init__()

        self.config = config or AttentionModelConfig()
        self.data_config = data_config or DataConfig()

        # Input dimension: n_subcarriers * 2 (real and imaginary)
        input_dim = self.data_config.N_SUBCARRIERS * 2

        # Core transformer
        self.transformer = TemporalAttentionTransformer(
            input_dim=input_dim,
            seq_len=self.data_config.SEQ_LEN,
            d_model=self.config.D_MODEL,
            n_heads=self.config.N_HEADS,
            n_layers=self.config.N_LAYERS,
            d_ff=self.config.FF_DIM,
            dropout=self.config.DROPOUT,
            n_classes=self.data_config.N_MCS_CLASSES,
            max_pos_encoding=100
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: [batch, seq_len, n_subcarriers, 2] complex channel coefficients
            attn_mask: Optional attention mask
            return_attn: If True, also returns attention weights
        Returns:
            logits: [batch, n_mcs_classes] (pre-softmax)
            If return_attn: (logits, attn_weights_list)
        """
        return self.transformer(x, attn_mask, return_attn)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted MCS index (hard decision)."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def get_attention_weights(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Get attention weights for visualization.
        Returns logits and list of attention weight tensors from all layers.
        """
        self.eval()
        return self.forward(x, attn_mask, return_attn=True)


def create_causal_attention_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device
) -> torch.Tensor:
    """Create causal (lower triangular) attention mask for autoregressive processing."""
    return create_attention_mask(seq_len, batch_size, device, mask_type="causal")


if __name__ == "__main__":
    # Test the model
    from training.config import set_global_seed
    from training.utils import count_parameters, measure_inference_latency

    set_global_seed(42)

    # Create model
    model = AttentionMCSModel().to(device)

    # Count parameters
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Test forward pass
    batch_size = 32
    seq_len = DataConfig.SEQ_LEN
    n_subcarriers = DataConfig.N_SUBCARRIERS

    dummy_input = torch.randn(batch_size, seq_len, n_subcarriers, 2, device=device)

    with torch.no_grad():
        logits = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {logits.shape}")

        # Test with attention retrieval
        logits, attn_weights = model.get_attention_weights(dummy_input)
        print(f"Number of attention layers: {len(attn_weights)}")
        print(f"Attention weights shape (layer 0): {attn_weights[0].shape}")

    # Measure inference latency
    print("\nMeasuring inference latency...")
    latency = measure_inference_latency(model, (batch_size, seq_len, n_subcarriers, 2), device=device)
    print(f"Average inference latency: {latency:.2f} ms")