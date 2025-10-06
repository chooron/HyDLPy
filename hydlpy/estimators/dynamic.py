from typing import List

import torch
import torch.nn as nn

from .base import DynamicEstimator


class LSTMEstimator(DynamicEstimator):
    """
    Estimator that uses an LSTM to predict dynamic parameters from time-series features.
    """

    def __init__(self, param_names: List[str], dynamic_input_dim: int, hidden_dim: int = 64, **kwargs):
        """
        Args:
            param_names (List[str]): A list of parameter names.
            dynamic_input_dim (int): Input dimension of the dynamic features.
            hidden_dim (int): Number of hidden units in the LSTM.
        """
        super().__init__(param_names=param_names)

        n_params = len(param_names)
        self.network = nn.LSTM(
            input_size=dynamic_input_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, n_params)

    def forward(self, dynamic_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Processes dynamic features to produce a tensor of dynamic parameters.
        """
        # LSTM expects (batch, seq_len, input_size).
        # Our data is (seq_len, batch, input_size), so permute.
        dynamic_features_permuted = dynamic_features.permute(1, 0, 2)

        lstm_out, _ = self.network(dynamic_features_permuted)

        # Reshape to (batch*seq_len, hidden_dim) to apply linear layer
        lstm_out_reshaped = lstm_out.reshape(-1, lstm_out.size(2))
        raw_params_reshaped = self.output_layer(lstm_out_reshaped)

        # Reshape back and permute to (seq_len, batch, n_params)
        n_timesteps = dynamic_features.size(0)
        n_basins = dynamic_features.size(1)
        raw_params = raw_params_reshaped.view(n_basins, n_timesteps, -1)
        raw_params = raw_params.permute(1, 0, 2)

        return raw_params
