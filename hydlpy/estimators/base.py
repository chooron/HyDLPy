from typing import Dict, List

import torch
import torch.nn as nn


class BaseEstimator(nn.Module):
    """
    Root abstract base class for all parameter estimators.
    """

    def __init__(
        self,
        hru_num: int,
        estimator: nn.Module,
        input_names: List[str],
        estimate_parameters: List[str],
    ):
        super().__init__()
        self._hru_num = hru_num
        self._estimator = estimator
        self._input_names = input_names
        self._estimate_parameters = estimate_parameters

    @property
    def input_size(self):
        return len(self._input_names)

    @property
    def estimator(self):
        return self._estimator

    @property
    def hru_num(self):
        return self._hru_num

    @property
    def estimate_parameters(self):
        return self._estimate_parameters

    @property
    def output_size(self):
        return self._hru_num, len(self._estimate_parameters)

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

    def __init__(
        self,
        hru_num: int,
        estimator: nn.Module,
        input_names: List[str],
        estimate_parameters: List[str],
    ):
        """
        Args:
            param_names (List[str]): A list of names for the parameters that
                this estimator will produce.
        """
        super().__init__(
            hru_num=hru_num,
            estimator=estimator,
            input_names=input_names,
            estimate_parameters=estimate_parameters,
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        # x -> basin num/ batch * input_dims
        output = self.estimator(x).view(x.shape[0], *self.output_size)
        return {k: v for k, v in zip(self.estimate_parameters, output.unbind(dim=-1))}


class DynamicEstimator(BaseEstimator):
    """
    Abstract base class for estimators of dynamic (time-varying) parameters.
    """

    def __init__(
        self,
        hru_num: int,
        estimator: nn.Module,
        input_names: List[str],
        estimate_parameters: List[str],
    ):
        """
        Args:
            param_names (List[str]): A list of names for the parameters that
                this estimator will produce.
        """
        super().__init__(
            hru_num=hru_num,
            estimator=estimator,
            input_names=input_names,
            estimate_parameters=estimate_parameters,
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        output = self.estimator(x).view(x.shape[0], x.shape[1], *self.output_size)
        return {k: v for k, v in zip(self.estimate_parameters, output.unbind(dim=-1))}
