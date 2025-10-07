# HyDLPy

<div align="center">

**🌊 A Python Framework for Building Hybrid Hydrological Models with Deep Learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/hydlpy.svg)](https://badge.fury.io/py/hydlpy)

*A Python implementation of the hybrid hydrological modeling framework inspired by [HydroModels.jl](https://github.com/chooron/HydroModels.jl)*

</div>

## 🚀 Overview

**HyDLPy** is a cutting-edge Python framework that seamlessly integrates traditional process-based hydrological models with modern deep learning techniques. Built on PyTorch and PyTorch Lightning, it enables researchers and practitioners to construct hybrid models that combine the interpretability of physics-based approaches with the flexibility and predictive power of neural networks.

### 🎯 Key Features

- **🔬 Symbolic Model Definition**: Define hydrological processes using SymPy equations with automatic compilation to PyTorch
- **🧠 Hybrid Architecture**: Combine process-based models with neural networks for enhanced performance
- **⚡ High Performance**: GPU-accelerated computations with automatic differentiation support
- **🔧 Modular Design**: Flexible, plug-and-play components for different modeling needs
- **📊 Built-in Models**: Pre-implemented HBV, XAJ, and ExpHydro models ready to use
- **🌐 Advanced Routing**: Sophisticated river routing with differentiable Muskingum-Cunge method
- **📈 Comprehensive Metrics**: Multiple loss functions and evaluation metrics for hydrological modeling
- **🔌 PyTorch Lightning Integration**: Easy training, validation, and deployment workflows

## 🏗️ Architecture

HyDLPy follows a modular architecture that allows for flexible combination of different components:

```mermaid
graph TB
    A[Input Data] --> B[Static Parameter Estimator]
    A --> C[Dynamic Parameter Estimator]
    A --> D[Hydrology Core]
    B --> D
    C --> D
    D --> E[Routing Module]
    E --> F[Streamflow Prediction]
    
    subgraph "Neural Components"
        B
        C
    end
    
    subgraph "Physics-Based Core"
        D
    end
    
    subgraph "Routing"
        E
    end
```

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
from hydlpy.hydrology.implements import HBV

# Create a simple configuration
config = {
    "hydrology": {
        "type": "hbv",
        "hidden_size": 1
    },
    "routing": {
        "type": "mean"
    },
    "optimizer": {
        "lr": 1e-3
    }
}

# Initialize the model
model = DplHydroModel(config)

# Prepare sample data
batch = {
    "x_forcing": {
        "P": torch.randn(1, 1, 100),  # Precipitation
        "T": torch.randn(1, 1, 100),  # Temperature
        "Ep": torch.randn(1, 1, 100)  # Potential evapotranspiration
    },
    "y": torch.randn(1, 1, 100)  # Target streamflow
}

# Run forward pass
with torch.no_grad():
    prediction = model(batch)
    print(f"Predicted streamflow shape: {prediction['y'].shape}")
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
model = HydrologicalModel(fluxes=fluxes, dfluxes=dfluxes, hidden_size=16)

# Run the model
states = torch.randn(16, 2)  # snowpack, soilwater
forcings = torch.randn(16, 2)  # temp, prcp

fluxes_out, new_states = model(forcings, states)
print(f"Fluxes shape: {fluxes_out.shape}")
print(f"New states shape: {new_states.shape}")
```

### Training with PyTorch Lightning

```python
import pytorch_lightning as pl
from hydlpy.data import HydroDataModule

# Create data module
data_module = HydroDataModule(
    data_path="path/to/your/data",
    batch_size=32,
    sequence_length=365
)

# Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

# Train the model
trainer.fit(model, data_module)
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

## 🛠️ Advanced Features

### Custom Loss Functions

```python
from hydlpy.criterion import KGEBatchLoss, NSEBatchLoss

# Use Kling-Gupta Efficiency loss
kge_loss = KGEBatchLoss()

# Use Nash-Sutcliffe Efficiency loss
nse_loss = NSEBatchLoss()
```

### Parameter Estimation

```python
# Static parameter estimation from basin characteristics
static_config = {
    "type": "mlp",
    "input_size": 10,  # Number of basin attributes
    "hidden_sizes": [64, 32],
    "param_names": ["FC", "BETA", "k0"]
}

# Dynamic parameter estimation from meteorological data
dynamic_config = {
    "type": "lstm",
    "input_size": 3,  # P, T, Ep
    "hidden_size": 32,
    "param_names": ["CFMAX", "TT"]
}
```

### Advanced Routing

```python
# Differentiable Muskingum-Cunge routing
routing_config = {
    "type": "dmc",
    "variables": {
        "p": 0.6,  # Wetted perimeter parameter
        "t": 86400,  # Time step in seconds
        "x": 0.2  # Storage coefficient
    }
}
```

## 📚 Documentation

For detailed documentation, examples, and API reference, visit:

- 📖 [Full Documentation](https://hydlpy.readthedocs.io)
- 🔬 [Model Building Guide](docs/build_hydrological_model.md)
- 💡 [Examples and Tutorials](examples/)

## 🤝 Contributing

We welcome contributions to HyDLPy! Here's how you can help:

### 🐛 Bug Reports

- Use the GitHub issue tracker
- Provide detailed reproduction steps
- Include system information and error messages

### 💡 Feature Requests

- Open an issue with the "enhancement" label
- Describe the use case and expected behavior
- Consider contributing the implementation

### 🔧 Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### 📝 Documentation

- Improve existing documentation
- Add examples and tutorials
- Fix typos and clarify explanations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
