from typing import Dict

import torch
import torch.nn as nn


class BaseRouting(nn.Module):
    """
    routing 需要考虑两种数据情况
    1. 考虑有拓扑关系的输入,比如河网,流向矩阵等
    2. 不考虑拓扑关系的输入,直接输出结果
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class MeanRouting(nn.Module):
    """
    Simplest routing: assumes runoff from the core model is already aggregated
    and just passes it through. If runoff is spatially distributed, it averages.
    """
    def __init__(self, **config):
        super().__init__()
        self.subbasin_areas = config.get('subbasin_areas')
        if self.subbasin_areas:
            self.area_coeff = torch.tensor(self.subbasin_areas) * 1000 / 3600
        else:
            self.area_coeff = 1.0

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        runoff = x['q_sim']
        if isinstance(self.area_coeff, torch.Tensor):
            self.area_coeff = self.area_coeff.to(runoff.device)
            runoff = runoff * self.area_coeff
        
        if runoff.dim() > 1 and runoff.shape[-1] > 1:
            return {'y': torch.sum(runoff, dim=-1, keepdim=True)}
        else:
            return {'y': runoff}

class MLPRouting(nn.Module):
    """
    Post-processes runoff with a simple MLP.
    """
    def __init__(self, **config):
        super().__init__()
        input_size = config.get('input_size', 10)
        hidden_size = config.get('hidden_size', 32)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        runoff = x['q_sim']
        prediction = self.net(runoff)
        return {'y': prediction}

class LSTMRouting(nn.Module):
    """
    Routes runoff using an LSTM layer.
    The input is expected to be a tensor of shape (batch_size, seq_len, n_features),
    where n_features is the number of sub-basins or reaches.
    """
    def __init__(self, **config):
        super().__init__()
        self.input_size = config.get('input_size', 11) # Should be n_mul
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0.0)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        runoff = x['q_sim'].unsqueeze(0)  # Shape: (batch_size, seq_len, input_size)

        # LSTM forward pass
        # The LSTM processes the sequence and returns output for each time step
        lstm_out, _ = self.lstm(runoff)  # lstm_out shape: (batch_size, seq_len, hidden_size)

        # Pass the output of each time step through the linear layer
        prediction = self.fc(lstm_out[0, :, :])  # prediction shape: (batch_size, seq_len, 1)

        return {'y': prediction}
