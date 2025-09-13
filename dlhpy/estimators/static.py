from typing import List

import torch
import torch.nn as nn

from .base import StaticEstimator


class DirectEstimator(StaticEstimator):
    """
    Estimator for static parameters treated as directly optimizable tensors.
    """

    def __init__(self, param_names: List[str], **kwargs):
        """
        Args:
            param_names (List[str]): A list of parameter names.
            n_basins (int): The number of basins.
        """
        super().__init__(param_names=param_names)

        n_params = len(param_names)
        self.params = nn.Parameter(torch.rand(kwargs.get("n_mul", 1), n_params))

    def forward(self, static_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns the stored parameters as a tensor."""
        return self.params


class MLPEstimator(StaticEstimator):
    """
    Estimator that uses an MLP to predict static parameters from static features.
    """

    def __init__(self, param_names: List[str], static_input_dim: int, hidden_dim: int = 64, **kwargs):
        """
        Args:
            param_names (List[str]): A list of parameter names.
            static_input_dim (int): Input dimension of the static features.
            hidden_dim (int): Number of hidden units in the MLP.
        """
        super().__init__(param_names=param_names)

        n_params = len(param_names)
        self.network = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_params),
        )

    def forward(self, static_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Processes static features to produce a tensor of static parameters."""
        return self.network(static_features)
