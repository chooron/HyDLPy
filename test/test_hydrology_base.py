import sympy
import torch.nn as nn
from dlhpy.hydrology_cores.base import HydrologyModel

class SimpleBucketModel(HydrologyModel):
    # 1. Define static parameter bounds
    _parameter_bounds = {"k": (0.01, 1.0)}
    
    # 2. Define state variables
    _state_variables = ["S"]

    # 3. Define sympy symbols for all variables and parameters
    S, P, k = sympy.symbols("S P k")
    
    # 4. (Optional) Initialize learnable parameters if they are torch managed
    # In this new setup, we pass them in `forward`, so this isn't needed.

    # 5. Define the physics with decorators
    @hydroflux
    def runoff():
        return k * S

    @stateflux("S")
    def dS_dt():
        runoff = sympy.Symbol("runoff")
        return P - runoff