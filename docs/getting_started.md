# 快速上手

本节将带你用最少的步骤运行 HyDLPy 中的一个混合水文模型（ExpHydro）。

## 安装

```bash
pip install hydlpy
```

## 基础示例（ExpHydro）

```python
import torch
from hydlpy.model import DplHydroModel

config = {
    "hydrology_model": {
        "name": "exphydro",
        "input_names": ["prcp", "pet", "temp"],
    },
    # 估计参数（静态∪动态）必须与物理模型参数完全一致
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

## 常见问题
- 输入键名固定为：`x_phy`, `x_nn_norm`, `xc_nn_norm`, `c_nn_norm`。
- `input_names` 必须与 `x_phy`/`xc_nn_norm` 的最后一维一致。
- 估计参数集合（静态∪动态）必须等于物理模型参数集合，否则在构造模型时报错。


