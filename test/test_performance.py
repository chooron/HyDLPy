# pytest -s -q test/test_performance.py -k test_performance_compare
import time
from typing import Tuple

import torch
import pytest

from hydlpy.hydrology.implements.exphydro import ExpHydro
from hydlpy.hydrology.implements.exphydro_manual import ExpHydroManual
from hydlpy.estimators.implements.direct import DirectEstimator


def _make_random_io(
    model,
    T: int,
    B: int,
    H: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create random forcings, states, and parameters tensors that match the model's
    declared metadata ordering.
    """
    F = len(model.forcing_names)
    S = len(model.state_names)
    P = len(model.parameter_names)

    forcings = torch.randn(T, B, H, F, device=device)
    states = torch.zeros(B, H, S, device=device)
    # parameters in [0,1] → transformed to bounds inside the model
    parameters = torch.nn.Parameter(torch.rand(B, H, P, device=device), requires_grad=True)
    parameters = parameters.unsqueeze(0).repeat(T, 1, 1, 1)
    return forcings, states, parameters


def _time_forward(model, forcings, states, parameters, iters: int = 5) -> float:
    # warmup
    with torch.no_grad():
        _ = model(forcings, states, parameters)

    torch.cuda.synchronize() if forcings.is_cuda else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(forcings, states, parameters)
    torch.cuda.synchronize() if forcings.is_cuda else None
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def _time_forward_backward(model, forcings, states, parameters, iters: int = 3) -> float:
    # simple scalar loss on outputs
    def _loss_fn(fluxes, new_states):
        return (fluxes.mean() + new_states.mean())

    torch.cuda.synchronize() if forcings.is_cuda else None
    t0 = time.perf_counter()
    for _ in range(iters):
        fluxes, new_states = model(forcings, states, parameters)
        loss = _loss_fn(fluxes, new_states)
        loss.backward()
        # zero grads between iterations
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
    torch.cuda.synchronize() if forcings.is_cuda else None
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def _build_direct_estimator_and_config(model, device: torch.device):
    # 使用 DirectEstimator 作为静态参数估计器
    # - estimate_parameters 与模型参数名一致
    # - input_names 使用模拟属性名（与强迫区分开）
    attr_names = ["attr1", "attr2", "attr3"]
    estimator = DirectEstimator(
        hru_num=model.hru_num,
        input_names=attr_names,
        estimate_parameters=model.parameter_names,
    ).to(device)
    config = {
        "hydrology_model": {
            "name": model.__class__.__name__,
            "input_names": model.forcing_names,
        },
        "static_estimator": {
            "name": "direct",
            "estimate_parameters": model.parameter_names,
            "input_names": attr_names,
        },
    }
    return estimator, config


def _time_direct_estimator(model, B: int, device: torch.device, iters: int = 5) -> float:
    estimator, _ = _build_direct_estimator_and_config(model, device)
    X = torch.randn(B, estimator.input_size, device=device)

    # warmup
    with torch.no_grad():
        _ = estimator(X)

    torch.cuda.synchronize() if X.is_cuda else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = estimator(X)
    torch.cuda.synchronize() if X.is_cuda else None
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def test_performance_compare():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem sizes
    T, B, H = 64, 16, 4

    # Models
    sym_model = ExpHydro(hru_num=H).to(device)
    manual_model = ExpHydroManual(hru_num=H).to(device)

    # IO tensors
    sym_forcings, sym_states, sym_params = _make_random_io(sym_model, T, B, H, device)
    man_forcings, man_states, man_params = _make_random_io(manual_model, T, B, H, device)

    # Forward timings
    sym_fwd = _time_forward(sym_model, sym_forcings, sym_states, sym_params, iters=5)
    man_fwd = _time_forward(manual_model, man_forcings, man_states, man_params, iters=5)

    # Forward + backward timings
    sym_bwd = _time_forward_backward(sym_model, sym_forcings, sym_states, sym_params, iters=3)
    man_bwd = _time_forward_backward(manual_model, man_forcings, man_states, man_params, iters=3)

    # Direct estimator timings + 基本配置打印
    sym_estimator, sym_cfg = _build_direct_estimator_and_config(sym_model, device)
    man_estimator, man_cfg = _build_direct_estimator_and_config(manual_model, device)
    sym_est = _time_direct_estimator(sym_model, B=B, device=device, iters=10)
    man_est = _time_direct_estimator(manual_model, B=B, device=device, iters=10)

    print("\nPerformance (seconds per run):")
    print(f"  ExpHydro (Symbolic)   - forward: {sym_fwd:.6f}, forward+backward: {sym_bwd:.6f}, direct-est: {sym_est:.6f}")
    print(f"  ExpHydro (Manual)     - forward: {man_fwd:.6f}, forward+backward: {man_bwd:.6f}, direct-est: {man_est:.6f}")
    print("\nConfigs:")
    print(f"  SymModel cfg: {sym_cfg}")
    print(f"  ManModel cfg: {man_cfg}")

    # Basic sanity assertions (just to ensure the test runs, not strict perf gates)
    assert sym_fwd > 0 and man_fwd > 0
    assert sym_bwd > 0 and man_bwd > 0
    assert sym_est > 0 and man_est > 0



@pytest.mark.parametrize("model_kind", ["symbolic", "manual"])
def test_make_random_io_shapes(model_kind):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, B, H = 8, 2, 1
    model = ExpHydro(hru_num=H).to(device) if model_kind == "symbolic" else ExpHydroManual(hru_num=H).to(device)
    forcings, states, params = _make_random_io(model, T, B, H, device)
    assert forcings.shape == (T, B, H, len(model.forcing_names))
    assert states.shape == (B, H, len(model.state_names))
    assert params.shape == (T, B, H, len(model.parameter_names))


@pytest.mark.parametrize("model_kind", ["symbolic", "manual"])
def test_time_forward_positive(model_kind):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, B, H = 16, 4, 2
    model = ExpHydro(hru_num=H).to(device) if model_kind == "symbolic" else ExpHydroManual(hru_num=H).to(device)
    f, s, p = _make_random_io(model, T, B, H, device)
    t = _time_forward(model, f, s, p, iters=2)
    assert t > 0


@pytest.mark.parametrize("model_kind", ["symbolic", "manual"])
def test_time_forward_backward_positive(model_kind):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, B, H = 16, 4, 2
    model = ExpHydro(hru_num=H).to(device) if model_kind == "symbolic" else ExpHydroManual(hru_num=H).to(device)
    f, s, p = _make_random_io(model, T, B, H, device)
    t = _time_forward_backward(model, f, s, p, iters=2)
    assert t > 0


@pytest.mark.parametrize("model_kind", ["symbolic", "manual"])
def test_build_direct_estimator_and_config(model_kind):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = 1
    model = ExpHydro(hru_num=H).to(device) if model_kind == "symbolic" else ExpHydroManual(hru_num=H).to(device)
    est, cfg = _build_direct_estimator_and_config(model, device)
    assert est is not None
    assert "hydrology_model" in cfg and "static_estimator" in cfg
    assert cfg["static_estimator"]["estimate_parameters"] == model.parameter_names


@pytest.mark.parametrize("model_kind", ["symbolic", "manual"])
def test_time_direct_estimator_positive(model_kind):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H = 8, 2
    model = ExpHydro(hru_num=H).to(device) if model_kind == "symbolic" else ExpHydroManual(hru_num=H).to(device)
    t = _time_direct_estimator(model, B=B, device=device, iters=3)
    assert t > 0
