# Estimators

HyDLPy provides two types of parameter estimators:

- StaticEstimator: infers time-invariant parameters from basin attributes
- DynamicEstimator: infers time-varying parameters from meteorological time series

The union of their output parameter names must equal the hydrology model parameter set.

## Static Estimator

- Config fields:
  - `name`: `mlp` | `direct`
  - `estimate_parameters`: list of output parameter names
  - `input_names`: list of attribute inputs
- Input shape: `c_nn_norm` is `[B, C]`
- Output shape: each parameter is `[B, H]`, where `H=hru_num`

Example:
```python
static_cfg = {
    "name": "mlp",
    "estimate_parameters": ["Tmin", "Tmax", "Df", "Smax"],
    "input_names": ["attr1", "attr2", "attr3", "attr4", "attr5", "attr6"],
}
```

## Dynamic Estimator

- Config fields:
  - `name`: `lstm`
  - `estimate_parameters`: list of output parameter names
  - `input_names`: list of meteorological inputs
- Input shape: `xc_nn_norm` is `[T, B, F]`
- Output shape: each parameter is `[T, B, H]`

Example:
```python
dynamic_cfg = {
    "name": "lstm",
    "estimate_parameters": ["Qmax", "f"],
    "input_names": ["attr1", "attr2", "attr3"],
}
```

## FAQ
- Parameter names must align with hydrology names (static ∪ dynamic = hydrology set)
- Input dimensions must match `input_names`
