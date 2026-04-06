"""
Baseline models for MCS selection.
Includes DNN, CNN, LSTM, and CNN+LSTM (Paper 2 replication).
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add project root to Python path for absolute imports when running as script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.config import (
    DataConfig,
    DNNConfig,
    CNNConfig,
    LSTMConfig
)


class DNNMCSModel(nn.Module):
    """Fully connected deep neural network with residual connections."""
    def __init__(
        self,
        config: Optional[DNNConfig] = None,
        data_config: Optional[DataConfig] = None,
        **kwargs
    ):
        super().__init__()

        self.config = config or DNNConfig()
        self.data_config = data_config or DataConfig()

        # Flattened input dimension
        input_dim = self.data_config.SEQ_LEN * self.data_config.N_SUBCARRIERS * 2
        hidden_dims = self.config.HIDDEN_DIMS
        n_classes = self.data_config.N_MCS_CLASSES

        layers = []
        prev_dim = input_dim

        # Build hidden layers with residual connections
        for i, hidden_dim in enumerate(hidden_dims):
            # Residual connection if dimensions match
            if prev_dim == hidden_dim:
                layers.append(ResidualBlock(prev_dim, hidden_dim, self.config.DROPOUT))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.config.DROPOUT))
            prev_dim = hidden_dim

        # Classification head
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_subcarriers, 2]
        Returns:
            logits: [batch, n_classes]
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten
        x = self.features(x)
        logits = self.classifier(x)
        return logits


class ResidualBlock(nn.Module):
    """Residual block for DNN."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Projection shortcut if dimensions differ
        self.shortcut = None
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        return self.act(out + residual)


class CNNMCSModel(nn.Module):
    """1D CNN over subcarriers + temporal pooling."""
    def __init__(
        self,
        config: Optional[CNNConfig] = None,
        data_config: Optional[DataConfig] = None,
        **kwargs
    ):
        super().__init__()

        self.config = config or CNNConfig()
        self.data_config = data_config or DataConfig()

        seq_len = self.data_config.SEQ_LEN
        n_subcarriers = self.data_config.N_SUBCARRIERS
        channels = self.config.CHANNELS
        kernel_size = self.config.KERNEL_SIZE

        # Input: [batch, seq_len, n_subcarriers, 2]
        # Process: Apply 1D CNN over subcarrier dimension for each time step independently
        # We'll treat each OFDM symbol separately: [batch*seq_len, 2, n_subcarriers]

        self.conv_layers = nn.ModuleList()
        in_channels = 2  # real and imaginary

        for out_channels in channels:
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            self.conv_layers.append(nn.BatchNorm1d(out_channels))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout(self.config.DROPOUT))
            in_channels = out_channels

        self.conv_net = nn.Sequential(*self.conv_layers)

        # After convolutions: compute the output length
        conv_output_len = n_subcarriers  # 1D conv with padding maintains length

        # Global average pooling over subcarrier and channel dimensions
        # After conv: [batch*seq_len, out_channels, n_subcarriers]
        # Flatten to [batch*seq_len, out_channels * n_subcarriers]
        self.flat_features = channels[-1] * conv_output_len

        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(256, self.data_config.N_MCS_CLASSES)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_subcarriers, 2]
        Returns:
            logits: [batch, n_classes]
        """
        batch_size, seq_len, n_subcarriers, _ = x.shape

        # Reshape to process each OFDM symbol independently
        x = x.view(batch_size * seq_len, n_subcarriers, 2)  # [B*T, N, 2]
        x = x.transpose(1, 2)  # [B*T, 2, N] for Conv1d

        # Apply convolutional layers
        x = self.conv_net(x)  # [B*T, C_out, N]

        # Flatten channel and subcarrier dimensions
        x = x.view(batch_size * seq_len, -1)  # [B*T, C_out * N]

        # Now we have a feature per OFDM symbol. Aggregate across temporal dimension (mean)
        x = x.view(batch_size, seq_len, -1)  # [B, T, C_out*N]
        x = x.mean(dim=1)  # [B, C_out*N] - temporal average pooling

        # Classification
        logits = self.classifier(x)
        return logits


class LSTMMCSModel(nn.Module):
    """LSTM for temporal modeling of channel evolution."""
    def __init__(
        self,
        config: Optional[LSTMConfig] = None,
        data_config: Optional[DataConfig] = None,
        **kwargs
    ):
        super().__init__()

        self.config = config or LSTMConfig()
        self.data_config = data_config or DataConfig()

        input_dim = self.data_config.N_SUBCARRIERS * 2
        hidden_size = self.config.HIDDEN_SIZE
        num_layers = self.config.NUM_LAYERS
        dropout = self.config.DROPOUT if num_layers > 1 else 0
        bidirectional = self.config.BIDIRECTIONAL

        # LSTM processes the sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_size * (2 if bidirectional else 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.data_config.N_MCS_CLASSES)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_subcarriers, 2]
        Returns:
            logits: [batch, n_classes]
        """
        batch_size = x.size(0)
        # Flatten subcarriers: [batch, seq_len, n_subcarriers * 2]
        x = x.view(batch_size, self.data_config.SEQ_LEN, -1)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_size] (last timestep output)

        # Use the output at the last time step
        last_output = lstm_out[:, -1, :]  # [batch, hidden_size]

        # Classification
        logits = self.classifier(last_output)
        return logits


class CNNLSTMMCSModel(nn.Module):
    """
    CNN + LSTM hybrid (Paper 2 replication).
    1D CNN extracts spatial features from subcarriers, LSTM models temporal dynamics.
    """
    def __init__(
        self,
        cnn_config: Optional[CNNConfig] = None,
        lstm_config: Optional[LSTMConfig] = None,
        data_config: Optional[DataConfig] = None,
        **kwargs
    ):
        super().__init__()

        self.cnn_config = cnn_config or CNNConfig()
        self.lstm_config = lstm_config or LSTMConfig()
        self.data_config = data_config or DataConfig()

        n_subcarriers = self.data_config.N_SUBCARRIERS

        # CNN feature extractor (same as CNNMCSModel but without classifier)
        channels = self.cnn_config.CHANNELS
        kernel_size = self.cnn_config.KERNEL_SIZE

        self.conv_layers = nn.ModuleList()
        in_channels = 2  # real and imaginary

        for out_channels in channels:
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            self.conv_layers.append(nn.BatchNorm1d(out_channels))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout(self.cnn_config.DROPOUT))
            in_channels = out_channels

        self.conv_net = nn.Sequential(*self.conv_layers)

        # Compute CNN output feature dimension
        conv_output_len = n_subcarriers  # length maintained due to padding
        cnn_output_dim = channels[-1] * conv_output_len

        # LSTM over temporal sequence of CNN features
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=self.lstm_config.HIDDEN_SIZE,
            num_layers=self.lstm_config.NUM_LAYERS,
            batch_first=True,
            dropout=self.lstm_config.DROPOUT if self.lstm_config.NUM_LAYERS > 1 else 0,
            bidirectional=self.lstm_config.BIDIRECTIONAL
        )

        lstm_output_dim = self.lstm_config.HIDDEN_SIZE * (2 if self.lstm_config.BIDIRECTIONAL else 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.data_config.N_MCS_CLASSES)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_subcarriers, 2]
        Returns:
            logits: [batch, n_classes]
        """
        batch_size, seq_len, n_subcarriers, _ = x.shape

        # Apply CNN to each time step independently
        # Reshape: [batch*seq_len, n_subcarriers, 2] -> [batch*seq_len, 2, n_subcarriers]
        x = x.view(batch_size * seq_len, n_subcarriers, 2)
        x = x.transpose(1, 2)

        # CNN feature extraction
        x = self.conv_net(x)  # [batch*seq_len, C_out, n_subcarriers]

        # Flatten CNN features
        x = x.view(batch_size * seq_len, -1)  # [batch*seq_len, C_out * n_subcarriers]

        # Reshape back to sequence for LSTM
        x = x.view(batch_size, seq_len, -1)  # [batch, seq_len, cnn_output_dim]

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Take last time step

        # Classification
        logits = self.classifier(last_output)
        return logits


# Registry for easy model creation
# AttentionMCSModel is imported at runtime to avoid circular imports
def _get_attention_model():
    from models.attention_mcs import AttentionMCSModel
    return AttentionMCSModel

MODEL_REGISTRY = {
    "attention": _get_attention_model(),
    "dnn": DNNMCSModel,
    "cnn": CNNMCSModel,
    "lstm": LSTMMCSModel,
    "cnn_lstm": CNNLSTMMCSModel,
}


def get_model(model_name: str, **kwargs):
    """Factory function to get model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)


if __name__ == "__main__":
    from training.config import set_global_seed
    from training.utils import count_parameters
    set_global_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test all baseline models
    batch_size = 16
    seq_len = 10
    n_subcarriers = 72

    test_input = torch.randn(batch_size, seq_len, n_subcarriers, 2).to(device)

    print("\n" + "="*60)
    print("Testing Baseline Models")
    print("="*60)

    for model_name in ["dnn", "cnn", "lstm", "cnn_lstm"]:
        model = get_model(model_name).to(device)
        n_params = count_parameters(model)
        with torch.no_grad():
            output = model(test_input)
        print(f"\n{model_name.upper()}:")
        print(f"  Parameters: {n_params:,}")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")

    # Test attention model
    print("\n" + "="*60)
    print("Testing Attention Model")
    print("="*60)
    attention_model = get_model("attention").to(device)
    n_params = count_parameters(attention_model)
    with torch.no_grad():
        logits, attn_weights = attention_model.get_attention_weights(test_input)
    print(f"\nATTENTION:")
    print(f"  Parameters: {n_params:,}")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Attention layers: {len(attn_weights)}")
    print(f"  Attention shape (layer 0): {attn_weights[0].shape}")