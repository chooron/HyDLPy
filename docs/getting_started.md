# Getting Started

This section walks you through the minimal steps to run a hybrid hydrological model (ExpHydro) in HyDLPy.

## Installation

```bash
pip install hydlpy
```

## Basic Example (ExpHydro)

```python
import torch
from hydlpy.model import DplHydroModel

config = {
    "hydrology_model": {
        "name": "exphydro",
        "input_names": ["prcp", "pet", "temp"],
    },
    # The union of static and dynamic estimated parameters must exactly match the hydrology parameters
    "static_estimator": {
        "name": "mlp",
        "estimate_parameters": ["Tmin", "Tmax", "Df", "Smax"],
        "input_names": ["attr1", "attr2", "attr3", "attr4", "attr5", "attr6"],
    },
    "dynamic_estimator": {
        "name": "lstm",
        "estimate_parameters": ["Qmax", "f"],
        "input_names": ["attr1", "attr2", "attr3"],
    },
    "warm_up": 100,
    "hru_num": 8,
}

model = DplHydroModel(config)

time_len, basin_num = 200, 20
batch = {
    "x_phy": torch.rand((time_len, basin_num, 3)),
    "x_nn_norm": torch.rand((time_len, basin_num, 3)),
    "xc_nn_norm": torch.rand((time_len, basin_num, 3)),
    "c_nn_norm": torch.rand((basin_num, 6)),
}

with torch.no_grad():
    outputs = model(batch)
    print("keys:", list(outputs.keys())[:5])
```

## FAQ
- Required input keys: `x_phy`, `x_nn_norm`, `xc_nn_norm`, `c_nn_norm`.
- `input_names` must match the last dimension of `x_phy`/`xc_nn_norm`.
- The (static ∪ dynamic) estimated parameters must equal the hydrology parameter set, otherwise the model constructor will raise an error.


