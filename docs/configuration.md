# Configuration and Input Conventions

This document describes the configuration fields of `DplHydroModel` and the input tensor format conventions.

## Top-level Configuration

```yaml
hydrology_model:
  name: exphydro | hbv
  input_names: [prcp, pet, temp]   # Must match the last dim of x_phy/xc_nn_norm

static_estimator:
  name: mlp | direct
  estimate_parameters: [..]        # Static parameter names
  input_names: [attr1, attr2, ...] # Static estimator inputs

dynamic_estimator:
  name: lstm
  estimate_parameters: [..]        # Dynamic parameter names
  input_names: [..]                # Dynamic estimator inputs

warm_up: 100
hru_num: 8
optimizer:
  lr: 1e-3
```

Note: The union of static and dynamic estimated parameters must exactly match the hydrology model parameter set, otherwise model construction will fail.

## Input Tensor Shapes

- `x_phy`: `[T, B, F]` hydrology core forcings (`F=len(hydrology_model.input_names)`)
- `x_nn_norm`: `[T, B, F]` reserved (same shape)
- `xc_nn_norm`: `[T, B, F]` dynamic estimator inputs (`F=len(dynamic_estimator.input_names)`)
- `c_nn_norm`: `[B, C]` static estimator inputs (`C=len(static_estimator.input_names)`)

## Example (ExpHydro)

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

## Example (HBV)

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

## Common Pitfalls
- Incomplete estimated parameter set: ensure static ∪ dynamic equals the hydrology parameter names
- Dimension mismatch: `input_names` must match the last dimension of inputs
- Missing `hru_num` or `warm_up`
