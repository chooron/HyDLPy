# HyDLPy

**🌊 A Python Framework for Building Hybrid Hydrological Models with Deep Learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/hydlpy.svg)](https://badge.fury.io/py/hydlpy)

*A Python implementation of the hybrid hydrological modeling framework inspired by [HydroModels.jl](https://github.com/chooron/HydroModels.jl)*


## 🚀 Overview

**HyDLPy** is a cutting-edge Python framework that seamlessly integrates traditional process-based hydrological models with modern deep learning techniques. Built on PyTorch and PyTorch Lightning, it enables researchers and practitioners to construct hybrid models that combine the interpretability of physics-based approaches with the flexibility and predictive power of neural networks.

**⚠️ WARNING: This is a private library for internal use only. It is highly unstable and the API is subject to change without notice. DO NOT USE IN PRODUCTION.**

### 🎯 Key Features

- **🔬 Symbolic Model Definition**: Define hydrological processes using SymPy equations with automatic compilation to PyTorch
- **🧠 Hybrid Architecture**: Combine process-based models with neural networks for enhanced performance
- **⚡ High Performance**: GPU-accelerated computations with automatic differentiation support
- **🔧 Modular Design**: Flexible, plug-and-play components for different modeling needs
- **📊 Built-in Models**: Pre-implemented HBV, XAJ, and ExpHydro models ready to use
- **🌐 Advanced Routing**: Sophisticated river routing with differentiable Muskingum-Cunge method
- **📈 Comprehensive Metrics**: Multiple loss functions and evaluation metrics for hydrological modeling
- **🔌 PyTorch Lightning Integration**: Easy training, validation, and deployment workflows

### Core Components

#### 🔬 **Hydrology Core**

The heart of the framework - a differentiable, physics-based hydrological model defined using symbolic equations:

- **Symbolic Definition**: Use SymPy to define complex hydrological processes
- **Automatic Compilation**: Equations are automatically compiled to optimized PyTorch functions
- **Topological Sorting**: Dependencies are automatically resolved for correct computation order
- **Parameter Management**: Built-in parameter bounds and optimization support

#### 🧠 **Parameter Estimators**

Neural network components that learn to estimate model parameters:

- **Static Parameter Estimator**: MLPs that estimate parameters from basin characteristics
- **Dynamic Parameter Estimator**: LSTMs/GRUs that estimate time-varying parameters from meteorological data
- **Flexible Architecture**: Support for various neural network architectures

#### 🌊 **Routing Modules**

Advanced river routing capabilities:

- **Differentiable Muskingum-Cunge (DMC)**: Physics-based routing with automatic differentiation
- **Neural Routing**: MLP-based routing for complex river networks
- **Flexible Integration**: Easy to add custom routing methods

#### 📊 **Built-in Models**

Ready-to-use implementations of popular hydrological models:

- **HBV Model**: Complete implementation of the HBV conceptual model
- **XAJ Model**: Xinanjiang model for humid regions
- **ExpHydro**: Experimental model for research and development

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- PyTorch 2.8 or higher
- CUDA (optional, for GPU acceleration)

### Install from PyPI

```bash
pip install hydlpy
```

### Install from Source

```bash
git clone https://github.com/chooron/hydlpy.git
cd hydlpy
pip install -e .
```

### Dependencies

- `torch>=2.8` - PyTorch for deep learning computations
- `pytorch-lightning>=2.5` - High-level PyTorch wrapper for training
- `sympy>=1.14` - Symbolic mathematics for model definition
- `pydantic>=2.11` - Data validation and settings management

## 🚀 Quick Start

### Basic Usage

```python
import torch
from hydlpy.model import DplHydroModel

# Create configuration (ExpHydro example)
config = {
    "hydrology_model": {
        "name": "exphydro",
        "input_names": ["prcp", "pet", "temp"],
    },
    # Estimated parameters must exactly match hydrology model parameters (static ∪ dynamic)
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
    "optimizer": {"lr": 1e-3},
}

model = DplHydroModel(config)

# Prepare input data (shapes must match input_names)
time_len, basin_num = 200, 20
batch = {
    "x_phy": torch.rand((time_len, basin_num, 3)),      # [T, B, F]
    "x_nn_norm": torch.rand((time_len, basin_num, 3)),  # reserved
    "xc_nn_norm": torch.rand((time_len, basin_num, 3)), # inputs for dynamic estimator
    "c_nn_norm": torch.rand((basin_num, 6)),            # inputs for static estimator
}

with torch.no_grad():
    outputs = model(batch)
    # outputs is a dict of fluxes and states, e.g. outputs["flow"], outputs["soilwater"]
    print(list(outputs.keys())[:5])
```

### Building a Custom Hydrological Model

```python
from hydlpy.hydrology import HydrologicalModel, HydroParameter, HydroVariable, variables
from sympy import S, Min, Max, Eq, tanh

# Define parameters with bounds
Tmin = HydroParameter("Tmin", default=-1.0, bounds=(-5.0, 5.0))
Smax = HydroParameter("Smax", default=250.0, bounds=(100.0, 400.0))

# Define variables
temp = HydroVariable("temp")
prcp = HydroVariable("prcp")
snowpack = HydroVariable("snowpack")
soilwater = HydroVariable("soilwater")

# Define intermediate fluxes
rainfall, snowfall, melt, flow = variables("rainfall, snowfall, melt, flow")

# Define equations
fluxes = [
    Eq(rainfall, (tanh(5.0 * (Tmin - temp)) + 1.0) * 0.5 * prcp),
    Eq(snowfall, (tanh(5.0 * (temp - Tmin)) + 1.0) * 0.5 * prcp),
    Eq(melt, (tanh(5.0 * (temp - Tmin)) + 1.0) * 0.5 * Min(snowpack, 2.5 * (temp - Tmin))),
    Eq(flow, Max(soilwater - Smax, 0))
]

dfluxes = [
    Eq(snowpack, snowfall - melt),
    Eq(soilwater, rainfall + melt - flow)
]

# Create the model
model = HydrologicalModel(fluxes=fluxes, dfluxes=dfluxes, hru_num=16)

# Run the model
states = torch.randn(16, 2)  # [H, S]
forcings = torch.randn(16, 2)  # [H, F]

fluxes_out, new_states = model(forcings.unsqueeze(0).unsqueeze(1), states)
print(f"Fluxes shape: {fluxes_out.shape}")
print(f"New states shape: {new_states.shape}")
```

### Training with PyTorch Lightning (optional)

```python
import pytorch_lightning as pl
from hydlpy.data import HydroDataModule

# Create data module
# data_module = HydroDataModule(...)

# Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

# Train the model
# trainer.fit(model, data_module)
```

## 📊 Built-in Models

HyDLPy comes with several pre-implemented hydrological models ready for immediate use:

### 🌨️ HBV Model

The HBV (Hydrologiska Byråns Vattenbalansavdelning) model is a conceptual rainfall-runoff model widely used in Nordic countries.

**Key Features:**

- Snow accumulation and melt processes
- Soil moisture accounting
- Three-layer response function
- Temperature-based snow/rain separation

**Parameters:**

- `TT`: Temperature threshold for snow/rain separation
- `CFMAX`: Degree-day factor for snowmelt
- `FC`: Field capacity of soil
- `BETA`: Shape parameter for soil moisture function
- `k0`, `k1`, `k2`: Recession coefficients

### 🌧️ XAJ Model

The Xinanjiang (XAJ) model is designed for humid regions and features a unique runoff generation mechanism.

**Key Features:**

- Three-layer soil moisture storage
- Free water storage concept
- Nash cascade routing
- Evapotranspiration calculation

**Parameters:**

- `Wum`, `Wlm`, `Wdm`: Upper, lower, and deep layer storage capacities
- `b`: Shape parameter for runoff generation
- `Smax`: Free water storage capacity
- `Ki`, `Kg`: Interflow and groundwater recession coefficients

### 🔬 ExpHydro Model

An experimental model designed for research and development purposes.

**Key Features:**

- Simplified snow processes
- Basic soil moisture accounting
- Configurable parameter bounds
- Ideal for testing new concepts


## 📚 Documentation

For detailed documentation, examples, and API reference, visit:

- 📖 [Full Documentation](https://hydlpy.readthedocs.io)
- 🔬 [Model Building Guide](build_hydrological_model.md)

## 🙏 Acknowledgments

- **HydroModels.jl**: Original Julia implementation by [chooron](https://github.com/chooron/HydroModels.jl)
- **PyTorch Team**: For the excellent deep learning framework
- **SymPy Community**: For symbolic mathematics capabilities
- **Hydrological Research Community**: For decades of model development and validation

## 📞 Support

- 💬 [GitHub Discussions](https://github.com/chooron/hydlpy/discussions) - Community support and questions
- 🐛 [Issue Tracker](https://github.com/chooron/hydlpy/issues) - Bug reports and feature requests
- 📧 Email: [jingxin0107@qq.com](mailto:jingxin0107@qq.com)

---

<div align="center">

**⭐ If you find HyDLPy useful, please give it a star on GitHub! ⭐**

Made with ❤️ for the hydrological modeling community

</div>
