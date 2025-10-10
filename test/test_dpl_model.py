import torch
from hydlpy.model import DplHydroModel

config = {
    "hydrology_model": {
        "name": "exphydro",
        "input_names": ["prcp", "pet", "temp"],
    },
    "static_estimator": {
        "name": "mlp",
        "estimate_parameters": ["Tmin", "Tmax", "Df", "Smax"],
        "input_names": [
            "attr1",
            "attr2",
            "attr3",
            "attr4",
            "attr5",
            "attr6",
            "attr7",
            "attr8",
            "attr9",
            "attr10",
        ],
    },
    "dynamic_estimator": {
        "name": "lstm",
        "estimate_parameters": ["Qmax", "f"],
        "input_names": ["prcp", "pet", "temp"],
    },
    "warm_up": 365,
    "hru_num": 16,
}

model = DplHydroModel(config)

time_len, basin_num = 730, 100

input_data = {
    "x_phy": torch.rand((time_len, basin_num, 3)),
    "x_nn_norm": torch.rand((time_len, basin_num, 3)),
    "xc_nn_norm": torch.rand((time_len, basin_num, 3)),
    "c_nn_norm": torch.rand((basin_num, 10)),
}

output = model(input_data)
