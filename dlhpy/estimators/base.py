from typing import List

import torch
import torch.nn as nn


class BaseEstimator(nn.Module):
    """
    Root abstract base class for all parameter estimators.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        The forward pass should be implemented by all subclasses.

        Returns:
            torch.Tensor: A tensor of estimated parameters.
            - For StaticEstimator: shape (n_basins, n_params).
            - For DynamicEstimator: shape (n_timesteps, n_basins, n_params).
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class StaticEstimator(BaseEstimator):
    """
    Abstract base class for estimators of static (time-invariant) parameters.
    """

    def __init__(self, param_names: List[str]):
        """
        Args:
            param_names (List[str]): A list of names for the parameters that
                this estimator will produce.
        """
        super().__init__()
        self.param_names = param_names

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")


class DynamicEstimator(BaseEstimator):
    """
    Abstract base class for estimators of dynamic (time-varying) parameters.
    """

    def __init__(self, param_names: List[str]):
        """
        Args:
            param_names (List[str]): A list of names for the parameters that
                this estimator will produce.
        """
        super().__init__()
        self.param_names = param_names

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")
