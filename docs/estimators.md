# 估计器（Estimators）

HyDLPy 提供两类参数估计器：

- 静态估计器（StaticEstimator）：从流域属性推断时间不变的参数
- 动态估计器（DynamicEstimator）：从气象时间序列推断时间变化参数

两者输出的参数名集合与物理模型参数名集合的并集必须完全一致。

## 静态估计器

- 配置字段：
  - `name`: `mlp` | `direct`
  - `estimate_parameters`: 列表，输出参数名
  - `input_names`: 列表，输入属性名
- 输入形状：`c_nn_norm` 为 `[B, C]`
- 输出形状：每个参数为 `[B, H]`，其中 `H=hru_num`

示例：
```python
static_cfg = {
    "name": "mlp",
    "estimate_parameters": ["Tmin", "Tmax", "Df", "Smax"],
    "input_names": ["attr1", "attr2", "attr3", "attr4", "attr5", "attr6"],
}
```

## 动态估计器

- 配置字段：
  - `name`: `lstm`
  - `estimate_parameters`: 列表，输出参数名
  - `input_names`: 列表，输入气象名
- 输入形状：`xc_nn_norm` 为 `[T, B, F]`
- 输出形状：每个参数为 `[T, B, H]`

示例：
```python
dynamic_cfg = {
    "name": "lstm",
    "estimate_parameters": ["Qmax", "f"],
    "input_names": ["attr1", "attr2", "attr3"],
}
```

## 常见问题
- 估计器的参数名请与物理模型参数名严格对齐（静态∪动态=物理参数集合）
- 输入维度需与 `input_names` 一致
