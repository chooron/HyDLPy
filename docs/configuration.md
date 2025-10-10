# 配置与输入约定

本文档说明 DplHydroModel 的配置字段及输入数据格式约定。

## 顶层配置

```yaml
hydrology_model:
  name: exphydro | hbv
  input_names: [prcp, pet, temp]   # 与 x_phy/xc_nn_norm 的最后一维严格对应

static_estimator:
  name: mlp | direct
  estimate_parameters: [..]        # 静态参数集合
  input_names: [attr1, attr2, ...] # 静态估计器输入

dynamic_estimator:
  name: lstm
  estimate_parameters: [..]        # 动态参数集合
  input_names: [..]                # 动态估计器输入

warm_up: 100
hru_num: 8
optimizer:
  lr: 1e-3
```

注意：静态与动态估计参数的并集必须与水文模型的参数集合完全一致，否则模型构造时报错。

## 输入张量形状

- `x_phy`: `[T, B, F]` 水文核心驱动输入，`F=len(hydrology_model.input_names)`
- `x_nn_norm`: `[T, B, F]` 预留（同形状）
- `xc_nn_norm`: `[T, B, F]` 动态估计器输入，`F=len(dynamic_estimator.input_names)`
- `c_nn_norm`: `[B, C]` 静态估计器输入，`C=len(static_estimator.input_names)`

## 示例（ExpHydro）

```python
config = {
    "hydrology_model": {"name": "exphydro", "input_names": ["prcp", "pet", "temp"]},
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
```

## 示例（HBV）

```python
config = {
    "hydrology_model": {"name": "hbv", "input_names": ["P", "Ep", "T"]},
    "static_estimator": {
        "name": "mlp",
        "estimate_parameters": [
            "TT", "CFMAX", "CWH", "CFR", "FC", "LP", "BETA", "k1", "k2", "UZL"
        ],
        "input_names": ["attr1", "attr2", "attr3", "attr4", "attr5", "attr6"],
    },
    "dynamic_estimator": {
        "name": "lstm",
        "estimate_parameters": ["BETA", "PPERC", "k0"],
        "input_names": ["attr1", "attr2", "attr3"],
    },
    "warm_up": 100,
    "hru_num": 8,
}
```

## 常见错误
- 估计参数集合不完整：请确保静态∪动态 = 物理参数名集合
- 输入维度不匹配：`input_names` 必须与对应输入最后一维一致
- 忘记提供 `hru_num` 或 `warm_up`
