from typing import List
import torch
import torch.nn as nn

from ..base import StaticEstimator


class DirectModel(nn.Module):
    def __init__(
        self,
        output_size: int,
    ) -> None:
        super().__init__()
        self.ouput_size = output_size
        self.model_params = nn.Parameter(torch.rand(output_size), requires_grad=True)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model_params.unsqueeze(0).repeat(x.shape[0], 1)
        return self.sigmoid_layer(out)


class DirectEstimator(StaticEstimator):
    def __init__(
        self,
        hru_num: int,
        input_names: List[str],
        estimate_parameters: List[str],
        **kwargs,
    ):
        estimator = DirectModel(
            output_size=len(estimate_parameters) * hru_num,
        )
        super().__init__(hru_num, estimator, input_names, estimate_parameters)
