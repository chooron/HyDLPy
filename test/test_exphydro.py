import torch
from hydlpy.hydrology.implements import ExpHydro

model = ExpHydro(hru_num=10)
input_data = torch.rand(365, 12, 3)
output, states = model(input_data)
print(output.shape)
