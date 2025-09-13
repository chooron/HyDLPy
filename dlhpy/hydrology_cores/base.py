import sympy
import inspect
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict
from graphlib import TopologicalSorter
from rich.console import Console
from rich.table import Table


def stateflux(func) -> staticmethod:
    func._is_stateflux = True
    return staticmethod(func)

def hydroflux(func) -> staticmethod:
    func._is_hydroflux = True
    return staticmethod(func)


class HydrologyModelMetaclass(type(nn.Module)):
    # 我们将主要逻辑从 __new__ 移到 __init__
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        state_variables_names = getattr(cls, '_state_variables', [])

        all_formulas_raw = OrderedDict()
        for attr_name, attr_value in attrs.items():
            # 关键：检查属性是否是 staticmethod
            if isinstance(attr_value, staticmethod):
                # 获取被包装的原始函数
                func_to_check = attr_value.__func__

                if hasattr(func_to_check, '_is_hydroflux') or hasattr(func_to_check, '_is_stateflux'):
                    all_formulas_raw[attr_name] = func_to_check

        if not all_formulas_raw:
            return

        print(f"[DLHPy Metaclass] Initializing model '{name}'...")

        # 1. 全自动创建符号
        all_symbol_names = set()
        all_symbol_names.update(all_formulas_raw.keys())
        for func in all_formulas_raw.values():
            all_symbol_names.update(inspect.signature(func).parameters.keys())

        symbols = {s_name: sympy.Symbol(s_name) for s_name in all_symbol_names}

        # 2. 使用创建的符号来构建表达式
        all_formulas = OrderedDict()
        for fname, func in all_formulas_raw.items():
            param_names = list(inspect.signature(func).parameters.keys())
            param_symbols = [symbols[p_name] for p_name in param_names]
            expr = func(*param_symbols)
            all_formulas[fname] = expr

        # 3. 后续所有逻辑完全不变
        calculated_variable_names = set(all_formulas.keys())
        state_variable_names_set = set(state_variables_names)
        dependency_graph = {
            fname: {str(s) for s in expr.free_symbols if
                    s.name in calculated_variable_names and s.name not in state_variable_names_set}
            for fname, expr in all_formulas.items()
        }

        try:
            ts = TopologicalSorter(dependency_graph)
            sorted_formulas_names = tuple(ts.static_order())
        except Exception as e:
            raise ValueError(f"Circular dependency detected in formulas! Details: {e}") from e

        final_expressions = OrderedDict()
        for fname in sorted_formulas_names:
            substituted_expr = all_formulas[fname].subs(final_expressions)
            final_expressions[fname] = substituted_expr

        parameter_names = set(getattr(cls, "_parameter_bounds", {}).keys())
        total_required_symbols = {s for expr in final_expressions.values() for s in expr.free_symbols}
        state_symbols = {s for s in total_required_symbols if s.name in state_variables_names}
        param_symbols = {s for s in total_required_symbols if s.name in parameter_names}
        forcing_symbols = total_required_symbols - state_symbols - param_symbols
        sorted_states = sorted([s.name for s in state_symbols])
        sorted_params = sorted([s.name for s in param_symbols])
        sorted_forcings = sorted([s.name for s in forcing_symbols])
        flat_input_symbols_obj = sorted(list(total_required_symbols), key=str)
        output_expressions = [final_expressions[name] for name in sorted_formulas_names]
        core_lambda = sympy.lambdify(flat_input_symbols_obj, output_expressions, modules='torch')

        def _process_core(self, forcings_dict, states_dict, params_dict, n_timesteps):
            states = states_dict.copy()
            outputs_over_time = {name: [] for name in self._output_variables}
            for t in range(n_timesteps):
                current_inputs = {**states, **params_dict}
                for key, val in forcings_dict.items():
                    current_inputs[key] = val[t]
                args = [current_inputs[str(sym)] for sym in self._flat_input_symbols_obj]
                all_results_tuple = self.core_lambda(*args)
                step_results = dict(zip(self._output_variables, all_results_tuple))
                for name, value in step_results.items():
                    outputs_over_time[name].append(value)
                for state_name in self._state_variables:
                    change = step_results[state_name]
                    states[state_name] = (states[state_name] + change).clamp(min=self.nearzero)
            final_outputs = {key: torch.stack(val) for key, val in outputs_over_time.items()}
            return final_outputs, states

        setattr(cls, '_process_core', _process_core)
        setattr(cls, '_state_variables', sorted_states)
        setattr(cls, '_parameter_variables', sorted_params)
        setattr(cls, '_forcing_variables', sorted_forcings)
        setattr(cls, '_output_variables', sorted_formulas_names)
        setattr(cls, '_flat_input_symbols_obj', flat_input_symbols_obj)
        setattr(cls, 'core_lambda', core_lambda)

        console = Console()
        table = Table(title=f"DLHPy Model Summary: [bold cyan]{name}[/bold cyan]", show_header=True,
                      header_style="bold magenta")
        table.add_column("Category", style="dim", width=20)
        table.add_column("Variables")
        table.add_row("Forcing Variables", ", ".join(sorted_forcings) or "None")
        table.add_row("State Variables", ", ".join(sorted_states) or "None")
        table.add_row("Parameter Variables", ", ".join(sorted_params) or "None")
        table.add_row("Output Variables", ", ".join(sorted_formulas_names) or "None")
        table.add_row("[dim]Execution Order[/dim]", f"[dim]{' → '.join(sorted_formulas_names)}[/dim]")
        console.print(table)


class HydrologyModel(torch.nn.Module, metaclass=HydrologyModelMetaclass):
    """
    Base class for hydrology models in the DLHPy framework, integrating
    parameter handling with automated core logic generation.
    """

    # --- To be defined by the user in their specific model class ---
    _parameter_bounds: Dict[str, tuple[float, float]] = {}
    _state_variables: list[str] = []  # e.g., ['S1', 'S2']

    # You can define default config values here
    _default_config = {
        "nearzero": 1e-6,
        "n_mul": 1,
        "variables": ["prcp", "temp", "pet"],
        "warm_up": 0,
    }

    def __init__(self, **config) -> None:
        super().__init__()

        # Merge user config with defaults
        self.config = self._default_config | config

        self.nearzero = self.config["nearzero"]
        self.warm_up = self.config["warm_up"]

        # This allows parameters to be defined as torch.nn.Parameter in the subclass __init__
        self._parameter_names_static = list(self._parameter_bounds.keys())

    @property
    def parameter_names(self) -> list[str]:
        return self._parameter_names_static

    def _initialize_states(self, n_basins: int) -> Dict[str, torch.Tensor]:
        """Creates initial states as zero tensors based on _state_variables."""
        # A default implementation, can be overridden by user
        states = {}
        for state_name in self._state_variables:
            states[state_name] = torch.zeros(n_basins, device=self.device)
        return states

    def _denormalize_parameters(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Applies sigmoid activation and denormalizes parameters to their defined bounds.
        This version assumes params passed are for the static parameters defined in the model.
        """
        activated_params = torch.sigmoid(params)
        denormalized_params = {}
        for i, param_name in enumerate(self.parameter_names):
            lower_bound, upper_bound = self._parameter_bounds[param_name]

            if activated_params.dim() == 2:  # Static parameters (n_basins, n_params)
                param_tensor = lower_bound + (upper_bound - lower_bound) * activated_params[:, i]
            # You can extend this to handle dynamic parameters if they also come from a single tensor
            else:
                raise ValueError(
                    f"Unsupported parameter tensor dimension: {activated_params.dim()}"
                )
            denormalized_params[param_name] = param_tensor
        return denormalized_params

    def forward(
            self,
            x_dict: Dict[str, torch.Tensor],
            static_params_norm: torch.Tensor,  # Normalized static params
            dynamic_params: Dict[str, torch.Tensor] = {},
            initial_states: Dict[str, torch.Tensor] = {},
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for the model.
        The `_process_core` method is now automatically generated.
        """
        # Determine n_basins from an input tensor
        n_basins = x_dict["prcp"].shape[1]
        self.device = x_dict["prcp"].device

        # Denormalize the static parameters
        static_params = self._denormalize_parameters(static_params_norm)

        # Handle initial states and warmup
        if initial_states:
            states = initial_states
        else:
            states = self._initialize_states(n_basins)
            if self.warm_up > 0:
                with torch.no_grad():
                    x_warmup = {k: v[: self.warm_up] for k, v in x_dict.items()}
                    dyn_params_warmup = {k: v[: self.warm_up] for k, v in dynamic_params.items()}

                    states = self._process_core(
                        x_warmup,
                        states,
                        static_params,
                        dyn_params_warmup,
                        return_states=True,
                    )

        # Run the model for the main simulation period
        x_main = {k: v[self.warm_up:] for k, v in x_dict.items()}
        dyn_params_main = {k: v[self.warm_up:] for k, v in dynamic_params.items()}

        return self._process_core(
            x_main, states, static_params, dyn_params_main, return_states=False
        )
