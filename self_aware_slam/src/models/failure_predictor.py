"""
Task 3.1: Model Architectures for Failure Prediction

Three models for predicting SLAM failure probability:

Model A - MLP baseline: Feature sequence flattened, fed through FC layers
Model B - LSTM temporal: Feature Encoder → LSTM → Failure Prediction Head
Model C - Transformer: Feature Encoder → Temporal Transformer → Failure Head

All models output:
  - failure_probability (sigmoid)
  - predicted_pose_error (regression head)
"""

import torch
import torch.nn as nn
import math


class FailureMLP(nn.Module):
    """Model A: MLP baseline.

    Flattens the temporal window and passes through FC layers.
    """

    def __init__(self, n_features: int = 7, window_size: int = 10,
                 hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        input_dim = n_features * window_size
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.failure_head = nn.Linear(prev_dim, 1)
        self.error_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, window_size, n_features)
        Returns:
            failure_prob: (batch, 1) sigmoid probability
            pred_error: (batch, 1) predicted pose error
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        h = self.backbone(x_flat)
        failure_prob = torch.sigmoid(self.failure_head(h))
        pred_error = torch.relu(self.error_head(h))
        return failure_prob, pred_error


class FailureLSTM(nn.Module):
    """Model B: LSTM temporal model.

    Feature Encoder → LSTM → Failure Prediction Head
    """

    def __init__(self, n_features: int = 7, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.failure_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.error_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, window_size, n_features)
        Returns:
            failure_prob: (batch, 1)
            pred_error: (batch, 1)
        """
        # Encode features at each timestep
        encoded = self.feature_encoder(x)  # (batch, seq, hidden)

        # LSTM over temporal sequence
        lstm_out, (h_n, _) = self.lstm(encoded)
        # Use last hidden state
        h_last = lstm_out[:, -1, :]  # (batch, hidden)

        failure_prob = torch.sigmoid(self.failure_head(h_last))
        pred_error = torch.relu(self.error_head(h_last))
        return failure_prob, pred_error


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FailureTransformer(nn.Module):
    """Model C: Transformer-based temporal model.

    Feature Encoder → Temporal Transformer → Failure Head

    Transformer attention may highlight key signals such as
    reprojection error spikes.
    """

    def __init__(self, n_features: int = 7, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.failure_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.error_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, window_size, n_features)
        Returns:
            failure_prob: (batch, 1)
            pred_error: (batch, 1)
        """
        encoded = self.feature_encoder(x)  # (batch, seq, d_model)
        encoded = self.pos_encoder(encoded)

        transformer_out = self.transformer(encoded)  # (batch, seq, d_model)
        # Use last token output (like CLS in BERT)
        h_last = transformer_out[:, -1, :]

        failure_prob = torch.sigmoid(self.failure_head(h_last))
        pred_error = torch.relu(self.error_head(h_last))
        return failure_prob, pred_error


def build_model(config: dict) -> nn.Module:
    """Factory function to build model from config.

    Args:
        config: Full config dict

    Returns:
        PyTorch model
    """
    model_type = config['model']['type']
    n_features = len(config['features']['names'])
    window_size = config['temporal']['window_size']

    if model_type == 'mlp':
        mc = config['model']['mlp']
        return FailureMLP(
            n_features=n_features,
            window_size=window_size,
            hidden_dims=mc['hidden_dims'],
            dropout=mc['dropout'],
        )
    elif model_type == 'lstm':
        mc = config['model']['lstm']
        return FailureLSTM(
            n_features=n_features,
            hidden_dim=mc['hidden_dim'],
            num_layers=mc['num_layers'],
            dropout=mc['dropout'],
        )
    elif model_type == 'transformer':
        mc = config['model']['transformer']
        return FailureTransformer(
            n_features=n_features,
            d_model=mc['d_model'],
            nhead=mc['nhead'],
            num_layers=mc['num_layers'],
            dropout=mc['dropout'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
