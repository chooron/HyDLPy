import torch
import torch.nn as nn
import pandas as pd
import sys
sys.path.append('E:\PyCode\HyDLPy')

from hydlpy.hydrology.implements import ExpHydro

class MyModel(nn.Module):
    def __init__(self, hidden_size):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.hydromodel = ExpHydro(hidden_size=hidden_size)

    def forward(self, x):
        """
            x: time series * forcing num
        """
        x = x.unsqueeze(1).expand(x.shape[0], self.hidden_size, -1)
        new_states = self.hydromodel.get_initial_state()
        outputs = []
        for i in range(x.shape[0]):
            output, new_states = self.hydromodel(x[i, :, :], new_states)
            outputs.append(output)
        output_arr = torch.stack(outputs, dim=0)
        return output_arr

exphydro = ExpHydro()
df = pd.read_csv(r'E:\PyCode\HyDLPy\test\data\3604000.csv')[exphydro.forcing_names]
input_arr = torch.from_numpy(df.values)

model = MyModel(hidden_size=16)
output = model(input_arr)

