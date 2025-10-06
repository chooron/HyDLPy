# Documentation: `HydrologicalModel`

The `HydrologicalModel` is a powerful PyTorch `nn.Module` that programmatically compiles a process-based hydrological model from a set of symbolic equations defined with SymPy.

It enables researchers and practitioners to define complex, interdependent physical processes in a clear mathematical format. The class then automatically analyzes these equations, determines the correct computational order, and compiles them into a high-performance, parallelized PyTorch function suitable for modern machine learning workflows, including automatic differentiation and GPU acceleration.

-----

## 1\. Installation and Dependencies

Before using the model, ensure you have the necessary libraries installed.

```bash
pip install torch sympy pytest
```

  - **torch**: For all numerical computations and neural network functionalities.
  - **sympy**: For defining the symbolic equations of the model.
  - **pytest**: For running the unit tests (optional but recommended).

-----

## 2\. Defining a Model

Creating a `HydrologicalModel` instance involves three main steps: defining the symbols, writing the equations, and instantiating the class.

### Step 2.1: Define Symbols

First, you must define all variables and parameters of your model using the provided `HydroParameter` and `HydroVariable` classes.

  - **`HydroParameter`**: Represents a model parameter. It requires a `default` value and can optionally take `bounds` for optimization. Each parameter will be a trainable `nn.Parameter` within the model.
  - **`HydroVariable`**: Represents any non-parameter symbol, such as a **state variable** (e.g., `snowpack`), a **forcing variable** (e.g., `temp`), or an **intermediate flux**.

<!-- end list -->

```python
# Import necessary classes
from hydlpy.hydrology_cores import HydrologicalModel, HydroParameter, HydroVariable, variables
from sympy import S, Min, Max, exp, tanh
import sympy
import torch

# --- Define Parameters ---
Tmin = HydroParameter("Tmin", default=-1.0, bounds=(-5.0, 5.0))
Tmax = HydroParameter("Tmax", default=1.0, bounds=(-5.0, 5.0))
Df = HydroParameter("Df", default=2.5, bounds=(0.0, 10.0))
Smax = HydroParameter("Smax", default=250.0, bounds=(100.0, 400.0))
Qmax = HydroParameter("Qmax", default=10.0, bounds=(0.0, 50.0))
f = HydroParameter("f", default=0.05, bounds=(0.0, 0.2))

# --- Define State and Forcing Variables ---
temp = HydroVariable("temp")
prcp = HydroVariable("prcp")
lday = HydroVariable("lday")
snowpack = HydroVariable("snowpack")
soilwater = HydroVariable("soilwater")

# --- Define intermediate flux symbols ---
# The 'variables' helper can create multiple symbols from a string
rainfall, snowfall, melt, pet, evap, baseflow, surfaceflow, flow = variables(
    "rainfall, snowfall, melt, pet, evap, baseflow, surfaceflow, flow"
)
```

### Step 2.2: Write the Equations

Next, define the model's physical processes as a list of `sympy.Eq` objects. The `HydrologicalModel` will automatically determine the correct computational order, even if the equations are provided out of order.

**Note:** Use `sympy.S()` to wrap any numerical constants (like `29.8` or `17.3`) to ensure type compatibility during compilation.

```python
# Helper function for smooth transitions
def step_func(x):
    return (tanh(5.0 * x) + 1.0) * 0.5

# Define equations for intermediate fluxes
fluxes = [
    sympy.Eq(pet, S(29.8) * lday * 24 * 0.611 * exp((S(17.3) * temp) / (temp + 237.3)) / (temp + 273.2)),
    sympy.Eq(rainfall, step_func(Tmin - temp) * prcp),
    sympy.Eq(snowfall, step_func(temp - Tmax) * prcp),
    sympy.Eq(melt, step_func(temp - Tmax) * Min(snowpack, Df * (temp - Tmax))),
    sympy.Eq(evap, step_func(soilwater) * pet * Min(soilwater, Smax) / Smax),
    sympy.Eq(baseflow, step_func(soilwater) * Qmax * exp(-f * (Smax - Min(Smax, soilwater)))),
    sympy.Eq(surfaceflow, Max(soilwater, Smax) - Smax),
    sympy.Eq(flow, baseflow + surfaceflow),
]

# Define differential equations for state variables
dfluxes = [
    sympy.Eq(snowpack, snowfall - melt),
    sympy.Eq(soilwater, (rainfall + melt) - (evap + flow)),
]
```

### Step 2.3: Instantiate the Model

Finally, create an instance of the `HydrologicalModel`, passing the equations and the desired `hidden_size`. The `hidden_size` determines how many parallel computational units (e.g., Hydrologic Response Units or HRUs) the model will simulate.

```python
# Instantiate a model to simulate 16 HRUs in parallel
model = HydrologicalModel(fluxes=fluxes, dfluxes=dfluxes, hidden_size=16)

# The model is now compiled and ready to use!
print(f"Model initialized with {model.hidden_size} HRUs.")
print(f"State variables: {model.state_names}")
print(f"Forcing variables: {model.forcing_names}")
```

-----

## 3\. Running the Model

The `HydrologicalModel` instance is a standard `nn.Module` and is called via its `forward` method. It expects pure `torch.Tensor` inputs with a shape of `(batch_size, hidden_size, num_features)`.

### Input and Output Tensors

  - **`states` (Input)**: A tensor containing the current values of the state variables (e.g., `snowpack`, `soilwater`).
      - Shape: `(batch_size, hidden_size, num_states)`
  - **`forcings` (Input)**: A tensor containing the current values of the forcing variables (e.g., `temp`, `prcp`).
      - Shape: `(batch_size, hidden_size, num_forcings)`
  - **`output` (Return)**: A single tensor containing all calculated intermediate fluxes and the **updated** state variables for the next time step.
      - Shape: `(batch_size, hidden_size, num_outputs)`

The order of features in the tensors corresponds to the alphabetically sorted lists stored in `model.state_names`, `model.forcing_names`, and `model.output_names`.

### Example: Running a Single Time Step

```python
# Define model dimensions
BATCH_SIZE = 4
N_HRUs = 16

# Create an instance of the model
model = HydrologicalModel(fluxes=fluxes, dfluxes=dfluxes, hidden_size=N_HRUs)

# 1. Prepare random input tensors with the correct 3D shape
# The order of features in the last dimension must match the model's name lists
states_tensor = torch.rand(BATCH_SIZE, N_HRUs, len(model.state_names))
forcings_tensor = torch.rand(BATCH_SIZE, N_HRUs, len(model.forcing_names))

# 2. Run the forward pass
# This executes one time step of the model for all items in the batch and all HRUs
output_tensor = model(states_tensor, forcings_tensor)

# 3. Interpret the output
print(f"Input states shape: {states_tensor.shape}")
print(f"Output tensor shape: {output_tensor.shape}")

# Use the model's output_map to get the index of a specific variable
flow_idx = model.output_map['flow']
new_soilwater_idx = model.output_map['soilwater']

# Extract the flow for the first item in the batch across all 16 HRUs
batch_0_flow = output_tensor[0, :, flow_idx]

# Extract the updated soilwater for the first item in the batch
batch_0_new_soilwater = output_tensor[0, :, new_soilwater_idx]

print(f"Calculated flow for batch item 0 (all HRUs): {batch_0_flow.shape}")
print(f"Updated soilwater for batch item 0 (all HRUs): {batch_0_new_soilwater.shape}")
```