import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import deque
from sympy import Eq, Symbol
from sympy.utilities.lambdify import lambdify
from sympy.printing.pytorch import TorchPrinter

from .symbol_toolkit import (
    is_parameter,
    SympyFunction,
    create_nn_module_wrapper,
    TORCH_EXTEND_MODULE,
)
from .hydrological_model import HydrologicalModel


class HydroSymolicModel(HydrologicalModel):
    """
    负责基于 SymPy 的模型解析与编译：
    - 解析符号，识别状态、强迫、参数与通量
    - 进行依赖拓扑排序
    - 使用 lambdify 编译得到 flux_calculator/state_updater
    - 聚合名称与边界并交由核心引擎 `HydrologicalModel` 运行
    """

    def __init__(
        self,
        *,
        fluxes: List[Eq],
        dfluxes: List[Eq],
        nns: Optional[Dict[str, nn.Module]] = None,
        hru_num: int = 1,
        input_names: Optional[List[str]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._dtype = dtype
        self._nns = nns or {}
        self._lambdify_modules_map = TORCH_EXTEND_MODULE.copy()
        self._input_names = input_names

        # --- 解析符号集 ---
        state_symbols_set = {eq.lhs for eq in dfluxes}
        flux_symbols_set = {eq.lhs for eq in fluxes}
        symbolic_fluxes = {eq.lhs: eq.rhs for eq in fluxes}

        # 自定义 nn.Module 函数包装注册
        for func_name, module_instance in self._nns.items():
            wrapper_callable = create_nn_module_wrapper(module_instance)
            self._lambdify_modules_map[func_name] = wrapper_callable

        # 通量拓扑排序，保证依赖顺序
        self._flux_symbols = self._topologically_sort_fluxes(
            flux_symbols_set, symbolic_fluxes
        )

        # 收集所有符号，区分参数与强迫
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

        self._state_symbols: List[Symbol] = sorted(list(state_symbols_set), key=str)
        self._parameter_symbols: List[Symbol] = sorted(
            list(parameter_symbols_set), key=str
        )
        self._forcing_symbols: List[Symbol] = sorted(list(forcing_symbols_set), key=str)

        state_names = [s.name for s in self._state_symbols]
        parameter_names = [s.name for s in self._parameter_symbols]
        forcing_names = [s.name for s in self._forcing_symbols]
        if input_names is not None:
            assert set(input_names) == set(forcing_names), (
                f"the input names: {input_names} should be equal to forcing name: {forcing_names}"
            )

        flux_names = [s.name for s in self._flux_symbols]

        # 参数边界
        parameter_bounds: Dict[str, Tuple[float, float]] = {}
        for s in self._parameter_symbols:
            bounds = s.get_bounds() or (0.0, 1.0)
            parameter_bounds[s.name] = bounds

        # 编译计算函数
        flux_calculator_callable = SympyFunction(
            self._compile_flux_calculator(fluxes), "flux_calculator"
        )
        state_updater_callable = SympyFunction(
            self._compile_state_updater(dfluxes), "state_updater"
        )

        # 交由核心引擎
        super().__init__(
            hru_num=hru_num,
            dtype=dtype,
            state_names=state_names,
            forcing_names=forcing_names,
            parameter_names=parameter_names,
            flux_names=flux_names,
            parameter_bounds=parameter_bounds,
            flux_calculator=flux_calculator_callable,
            state_updater=state_updater_callable,
            input_names=input_names,
        )

    # -------------------- SymPy 构建辅助 --------------------
    def _topologically_sort_fluxes(
        self, flux_symbols_set: set, symbolic_fluxes: Dict
    ) -> List[Symbol]:
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

    def _compile_flux_calculator(self, fluxes: List[Eq]):
        symbolic_fluxes = {eq.lhs: eq.rhs for eq in fluxes}
        fully_substituted_fluxes = {}
        for flux_sym in self._flux_symbols:
            expression = symbolic_fluxes[flux_sym]
            substituted_expression = expression.subs(fully_substituted_fluxes)
            fully_substituted_fluxes[flux_sym] = substituted_expression

        final_flux_exprs = [fully_substituted_fluxes[s] for s in self._flux_symbols]
        input_symbols = [
            self._state_symbols,
            self._forcing_symbols,
            self._parameter_symbols,
        ]
        return lambdify(
            input_symbols,
            final_flux_exprs,
            cse=True if len(self._nns) == 0 else False,
            modules=[self._lambdify_modules_map, "torch"],
            printer=TorchPrinter({"strict": False}),
        )

    def _compile_state_updater(self, dfluxes: List[Eq]):
        symbolic_dfluxes = {eq.lhs: eq.rhs for eq in dfluxes}
        next_state_exprs = [s + symbolic_dfluxes.get(s, 0) for s in self._state_symbols]
        input_symbols = [
            self._state_symbols,
            self._flux_symbols,
            self._forcing_symbols,
            self._parameter_symbols,
        ]
        return lambdify(
            input_symbols,
            next_state_exprs,
            cse=True if len(self._nns) == 0 else False,
            modules=[self._lambdify_modules_map, "torch"],
            printer=TorchPrinter({"strict": False}),
        )
