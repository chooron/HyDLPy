import torch
import torch.nn as nn
from sympy import Eq, Symbol, Function
from sympy.utilities.lambdify import lambdify
from sympy.printing.pytorch import TorchPrinter
from typing import Callable, Dict, List, Optional, Tuple
from collections import deque
from .symbol_toolkit import is_parameter, SympyFunction


# Helper function to ensure inputs to min/max are Tensors
def _to_tensor(val, ref=None):
    if torch.is_tensor(val):
        return val
    device = ref.device if (ref is not None and torch.is_tensor(ref)) else None
    return torch.tensor(val, dtype=torch.float32, device=device)


# Custom module for lambdify to handle mixed-type min/max
TORCH_EXTEND_MODULE = {
    "Max": lambda a, b: torch.maximum(_to_tensor(a, b), _to_tensor(b, a)),
    "max": lambda a, b: torch.maximum(_to_tensor(a, b), _to_tensor(b, a)),
    "Min": lambda a, b: torch.minimum(_to_tensor(a, b), _to_tensor(b, a)),
    "min": lambda a, b: torch.minimum(_to_tensor(a, b), _to_tensor(b, a)),
}


def create_nn_module_wrapper(model: nn.Module) -> Callable:
    """
    Creates a wrapper function for an nn.Module to make it compatible
    with the multi-argument output of SymPy's lambdify.

    Args:
        model (nn.Module): The PyTorch module to wrap.

    Returns:
        Callable: A new function that accepts multiple tensor arguments,
                    stacks them, passes them to the model, and returns the result.
    """

    def wrapper(*args: torch.Tensor) -> torch.Tensor:
        """
        This inner function will be called by lambdify.
        It takes individual tensors, preprocesses them, and calls the nn.Module.
        """
        if not args:
            raise ValueError("Custom nn.Module function was called with no arguments.")

        device, dtype = args[0].device, args[0].dtype
        inputs = torch.stack([torch.atleast_1d(arg) for arg in args], dim=-1).to(
            device=device, dtype=dtype
        )

        return model(inputs).squeeze(-1)

    return wrapper


class HydrologicalModel(nn.Module):
    """
    A PyTorch nn.Module that programmatically compiles a hydrological model
    from a set of symbolic SymPy equations.
    """

    def __init__(
        self,
        fluxes: List[Eq],
        dfluxes: List[Eq],
        nns: Optional[Dict[str, nn.Module]] = None,
        hidden_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.flux_eqs = fluxes
        self.dflux_eqs = dfluxes
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.nns = nn.ModuleDict(nns or {})
        self.lambdify_modules_map = TORCH_EXTEND_MODULE.copy()

        self.state_symbols: List[Symbol] = []
        self.parameter_symbols: List[Symbol] = []
        self.forcing_symbols: List[Symbol] = []
        self.flux_symbols: List[Symbol] = []
        self.state_names: List[str] = []
        self.parameter_names: List[str] = []
        self.forcing_names: List[str] = []
        self.flux_names: List[str] = []
        self.parameter_bounds: Dict[str, Tuple[float, float]] = {}
        self.flux_map: Dict[str, int] = {}
        self.state_map: Dict[str, int] = {}

        self._initialize_model(fluxes, dfluxes)
        self._initialize_parameters()
        self.flux_calculator = SympyFunction(
            self._compile_flux_calculator(fluxes), "flux_calculator"
        )
        self.state_updater = SympyFunction(
            self._compile_state_updater(dfluxes), "state_updater"
        )

    def _topologically_sort_fluxes(
        self, flux_symbols_set: set, symbolic_fluxes: Dict
    ) -> List[Symbol]:
        """Topologically sorts flux equations to respect dependencies."""
        in_degree = {s: 0 for s in flux_symbols_set}
        graph = {s: [] for s in flux_symbols_set}
        for flux_var in flux_symbols_set:
            dependencies = symbolic_fluxes[flux_var].free_symbols
            for dep in dependencies:
                if dep in flux_symbols_set:
                    graph[dep].append(flux_var)
                    in_degree[flux_var] += 1
        queue = deque([s for s in flux_symbols_set if in_degree[s] == 0])
        sorted_fluxes = []
        while queue:
            u = queue.popleft()
            sorted_fluxes.append(u)
            for v in graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        if len(sorted_fluxes) != len(flux_symbols_set):
            raise ValueError(
                "A circular dependency was detected in the flux equations."
            )
        return sorted_fluxes

    def _initialize_model(self, fluxes: List[Eq], dfluxes: List[Eq]):
        """Identifies all symbols, creates wrappers for custom functions, and organizes them."""
        state_symbols_set = {eq.lhs for eq in dfluxes}
        flux_symbols_set = {eq.lhs for eq in fluxes}
        symbolic_fluxes = {eq.lhs: eq.rhs for eq in fluxes}

        all_custom_funcs = set()
        for eq in fluxes + dfluxes:
            all_custom_funcs.update(eq.rhs.atoms(Function))

        for func_name, module_instance in self.nns.items():
            wrapper_callable = create_nn_module_wrapper(module_instance)
            self.lambdify_modules_map[func_name] = wrapper_callable

        self.flux_symbols = self._topologically_sort_fluxes(
            flux_symbols_set, symbolic_fluxes
        )

        all_symbols = set()
        for eq in fluxes + dfluxes:
            all_symbols.update(eq.free_symbols)

        parameter_symbols_set, forcing_symbols_set = set(), set()
        for s in all_symbols:
            if s in state_symbols_set or s in flux_symbols_set:
                continue
            elif is_parameter(s):
                parameter_symbols_set.add(s)
            else:
                forcing_symbols_set.add(s)

        self.state_symbols = sorted(list(state_symbols_set), key=str)
        self.parameter_symbols = sorted(list(parameter_symbols_set), key=str)
        self.forcing_symbols = sorted(list(forcing_symbols_set), key=str)
        self.state_names = [s.name for s in self.state_symbols]
        self.parameter_names = [s.name for s in self.parameter_symbols]
        self.forcing_names = [s.name for s in self.forcing_symbols]
        self.flux_names = [s.name for s in self.flux_symbols]
        self.flux_map = {name: i for i, name in enumerate(self.flux_names)}
        self.state_map = {name: i for i, name in enumerate(self.state_names)}

    def _initialize_parameters(self):
        """Initializes trainable parameters and pre-computes boundary tensors."""
        min_bounds_list = []
        max_bounds_list = []
        for s in self.parameter_symbols:
            bounds = s.get_bounds() or (0.0, 1.0)
            initial_value = s.get_default()
            if initial_value is None:
                raise ValueError(
                    f"Parameter '{s.name}' must have a 'default' value provided."
                )
            self.parameter_bounds[s.name] = bounds
            min_b, max_b = bounds
            min_bounds_list.append(min_b)
            max_bounds_list.append(max_b)
            initial_tensor = torch.full(
                (self.hidden_size,), float(initial_value), dtype=self.dtype
            )
            unconstrained_values = torch.logit(
                (initial_tensor - min_b) / (max_b - min_b)
            )
            setattr(self, s.name, nn.Parameter(unconstrained_values))

        self.register_buffer(
            "min_bounds", torch.tensor(min_bounds_list, dtype=self.dtype)
        )
        self.register_buffer(
            "max_bounds", torch.tensor(max_bounds_list, dtype=self.dtype)
        )

    def _get_initial_state(self) -> torch.Tensor:
        """
        Returns the initial state of the model.

        The initial state is a PyTorch tensor of shape (hidden_size, num_state_variables).
        By default, all state variables are initialized to 0.0.

        Returns:
            torch.Tensor: A tensor representing the initial state.
        """
        num_state_variables = len(self.state_names)
        return torch.zeros(self.hidden_size, num_state_variables, dtype=self.dtype)

    def _compile_flux_calculator(self, fluxes: List[Eq]):
        """Compiles the flux equations into a callable PyTorch function."""
        symbolic_fluxes = {eq.lhs: eq.rhs for eq in fluxes}
        fully_substituted_fluxes = {}
        for flux_sym in self.flux_symbols:
            expression = symbolic_fluxes[flux_sym]
            substituted_expression = expression.subs(fully_substituted_fluxes)
            fully_substituted_fluxes[flux_sym] = substituted_expression

        final_flux_exprs = [fully_substituted_fluxes[s] for s in self.flux_symbols]
        input_symbols = [
            self.state_symbols,
            self.forcing_symbols,
            self.parameter_symbols,
        ]
        return lambdify(
            input_symbols,
            final_flux_exprs,
            modules=[self.lambdify_modules_map, "torch"],
            printer=TorchPrinter({"strict": False}),
        )

    def _compile_state_updater(self, dfluxes: List[Eq]):
        """Compiles the state update equations into a callable PyTorch function."""
        symbolic_dfluxes = {eq.lhs: eq.rhs for eq in dfluxes}
        next_state_exprs = [s + symbolic_dfluxes.get(s, 0) for s in self.state_symbols]
        input_symbols = [
            self.state_symbols,
            self.flux_symbols,
            self.forcing_symbols,
            self.parameter_symbols,
        ]

        return lambdify(
            input_symbols,
            next_state_exprs,
            modules=[self.lambdify_modules_map, "torch"],
            printer=TorchPrinter({"strict": False}),
        )

    def _transform_parameters(self, unconstrained_params: torch.Tensor) -> torch.Tensor:
        """Transforms a tensor of unconstrained parameters to their bounded physical values."""
        sigmoid_params = torch.sigmoid(unconstrained_params)
        return self.min_bounds + (self.max_bounds - self.min_bounds) * sigmoid_params

    def _core(
        self,
        forcings: torch.Tensor,
        states: torch.Tensor,
        parameters: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes one time step for multiple parallel units (HRUs).
        """
        forcing_values = forcings.unbind(-1)
        state_values = states.unbind(-1)
        param_values = parameters.unbind(-1)

        flux_outputs_tuple = self.flux_calculator(
            state_values, forcing_values, param_values
        )
        flux_outputs = torch.stack(flux_outputs_tuple, dim=-1)

        flux_values = flux_outputs.unbind(-1)
        new_states_tuple = self.state_updater(
            state_values, flux_values, forcing_values, param_values
        )
        new_states = torch.stack(new_states_tuple, dim=-1)

        torch.clamp_(new_states, min=0.0)

        return flux_outputs, new_states

    def _process_parameters(
        self,
        parameters: Optional[torch.Tensor],
        forcings_shape: Tuple[int, int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Validates, transforms, and prepares model parameters for the simulation loop.

        Args:
            parameters: An optional tensor of external parameters. If None, uses internal model parameters.
            timelen: The length of the time dimension for the simulation.
            device: The torch device (e.g., 'cpu' or 'cuda') to place new tensors on.

        Returns:
            A tensor of transformed parameters ready for the simulation, with a time dimension.
        """
        T, B, H, F = forcings_shape
        unconstrained_params: torch.Tensor
        if parameters is not None:
            expected_dynamic_shape = (T, B, H, len(self.parameter_names))
            expected_static_shape = (B, H, len(self.parameter_names))
            if (parameters.shape != expected_dynamic_shape) & (
                parameters.shape != expected_static_shape
            ):
                raise ValueError(
                    f"Provided parameters have shape {parameters.shape}, "
                    + f"but expected dynamic shape: {expected_dynamic_shape} or static shape: {expected_static_shape}"
                )
            unconstrained_params = parameters
        else:
            internal_params_list = [
                getattr(self, name) for name in self.parameter_names
            ]
            if not internal_params_list:
                # Create an empty tensor if there are no parameters
                unconstrained_params = torch.empty(self.hidden_size, 0, device=device)
            else:
                unconstrained_params = torch.stack(internal_params_list, dim=1)

        # Apply transformations (e.g., sigmoid) to ensure parameters are in a valid range
        transformed_parameters: torch.Tensor = self._transform_parameters(
            unconstrained_params
        )

        # Ensure parameters have a time dimension, repeating if they are static
        if len(transformed_parameters.shape) == 4:
            pass
        elif len(transformed_parameters.shape) == 3:
            transformed_parameters = transformed_parameters.unsqueeze(0).repeat(
                T, 1, 1, 1
            )
        return transformed_parameters

    def forward(
        self,
        forcings: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        parameters: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the full time-series simulation.

        Args:
            forcings: A tensor of input forcing data with shape [T, B, H, F],
                      where T=time, B=basins, H=HRUs, F=features.
            states: A tensor of the initial system states with shape [B, H, S],
                    where S is the number of states.
            parameters: An optional tensor of model parameters to override internal ones.

        Returns:
            A tuple containing:
            - torch.Tensor: The time series of calculated fluxes.
            - torch.Tensor: The time series of simulated states.
        """
        # time step, basin num, hru num, feature dim
        T, B, H, F = forcings.shape

        # Correctly get device from an existing tensor
        device = forcings.device

        transformed_parameters = self._process_parameters(parameters, forcings.shape, device)

        fluxes_placeholder = torch.zeros((T, B, H, len(self.flux_names)), device=device)
        states_placeholder = torch.zeros(
            (T, B, H, len(self.state_names)), device=device
        )

        if states is None:
            states = self._get_initial_state()
        current_states = torch.clone(states)

        for i in range(T):
            fluxes_, states_ = self._core(
                forcings[i, :, :, :], current_states, transformed_parameters[i, :, :, :]
            )
            fluxes_placeholder[i, :, :, :] = fluxes_
            states_placeholder[i, :, :, :] = states_
            current_states = states_  # Update states for the next iteration

        return fluxes_placeholder, states_placeholder
