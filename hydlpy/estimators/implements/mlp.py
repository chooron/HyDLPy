from typing import List
import torch
import torch.nn as nn

from ..base import StaticEstimator


class MlpModel(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.hidden_layer = nn.ModuleList(
            [
                nn.Linear(in_features=hidden_size, out_features=hidden_size)
                for _ in range(num_layers)
            ]
        )
        self.fc_layer = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout_layer(self.input_layer(x))
        for linear in self.hidden_layer:
            x = self.dropout_layer(linear(x))
        out = self.dropout_layer(self.fc_layer(x))
        return self.sigmoid_layer(out)


class MlpEstimator(StaticEstimator):
    def __init__(
        self,
        hru_num: int,
        input_names: List[str],
        estimate_parameters: List[str],
        **kwargs,
    ):
        estimator = MlpModel(
            input_size=len(input_names),
            output_size=len(estimate_parameters) * hru_num,
            hidden_size=kwargs.get("hidden_size", 128),
            num_layers=kwargs.get("num_layers", 1),
            dropout=kwargs.get("dropout", 0.0),
        )
        super().__init__(hru_num, estimator, input_names, estimate_parameters)
