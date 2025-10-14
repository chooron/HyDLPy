from typing import List
import torch
import torch.nn as nn

from ..base import DynamicEstimator


class GruModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu_layer = nn.ReLU()
        self.gru_layer = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.fc_layer = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.relu_layer(x)
        h, _ = self.gru_layer(x)
        out = self.fc_layer(h)
        return self.sigmoid_layer(out)


class GruEstimator(DynamicEstimator):
    def __init__(
        self,
        hru_num: int,
        input_names: List[str],
        estimate_parameters: List[str],
        **kwargs,
    ):
        estimator = GruModel(
            input_size=len(input_names),
            output_size=len(estimate_parameters) * hru_num,
            hidden_size=kwargs.get("hidden_size", 128),
            num_layers=kwargs.get("num_layers", 1),
            dropout=kwargs.get("dropout", 0.0),
        )
        super().__init__(hru_num, estimator, input_names, estimate_parameters)