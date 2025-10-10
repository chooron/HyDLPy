import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hydlpy.model import DplHydroModel  # noqa: E402


@pytest.fixture(scope="module")
def exphydro_config():
    """ExpHydro模型配置fixture"""
    return {
        "hydrology_model": {
            "name": "exphydro",
            "input_names": ["prcp", "pet", "temp"],
        },
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


@pytest.fixture(scope="module")
def hbv_config():
    """HBV模型配置fixture"""
    return {
        "hydrology_model": {
            "name": "hbv",
            "input_names": ["P", "Ep", "T"],
        },
        "static_estimator": {
            "name": "mlp",
            "estimate_parameters": [
                "TT",
                "CFMAX",
                "CWH",
                "CFR",
                "FC",
                "LP",
                "BETA",
                "k1",
                "k2",
                "UZL",
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


@pytest.fixture(scope="module")
def test_data():
    time_len, basin_num = 200, 20
    return {
        "x_phy": torch.rand((time_len, basin_num, 3)),
        "x_nn_norm": torch.rand((time_len, basin_num, 3)),
        "xc_nn_norm": torch.rand((time_len, basin_num, 3)),
        "c_nn_norm": torch.rand((basin_num, 6)),
    }


def test_exphydro_basic(exphydro_config, test_data):
    """测试ExpHydro基础功能"""
    print("\n--- Running Test: ExpHydro Basic ---")
    model = DplHydroModel(exphydro_config)
    output = model(test_data)
    model_output_names = model.hydrology_model.flux_names
    model_state_names = model.hydrology_model.state_names
    for flux in model_output_names:
        assert flux in output
        assert isinstance(output[flux], torch.Tensor)
    for state in model_state_names:
        assert state in output
        assert isinstance(output[state], torch.Tensor)

def test_exphydro_estimators_output_shapes(exphydro_config, test_data):
    """测试静态和动态参数估计器的输出形状（使用统一test_data）"""
    print("\n--- Running Test: Estimators Output Shapes ---")
    model = DplHydroModel(exphydro_config)
    basin_num = test_data["c_nn_norm"].shape[0]
    time_len = test_data["xc_nn_norm"].shape[0]

    # 静态估计器 (basin_num, hru_num)
    static_out = model.static_estimator(test_data["c_nn_norm"])
    for k, v in static_out.items():
        assert v.shape[0] == basin_num
        assert v.shape[1] == exphydro_config["hru_num"]

    # 动态估计器 (time_len, basin_num, hru_num)
    dynamic_out = model.dynamic_estimator(test_data["xc_nn_norm"])
    for k, v in dynamic_out.items():
        assert v.shape[0] == time_len
        assert v.shape[1] == basin_num
        assert v.shape[2] == exphydro_config["hru_num"]


def test_hbv_basic(hbv_config, test_data):
    """测试HBV基础功能"""
    print("\n--- Running Test: HBV Basic ---")
    model = DplHydroModel(hbv_config)
    output = model(test_data)
    model_output_names = model.hydrology_model.flux_names
    model_state_names = model.hydrology_model.state_names
    for flux in model_output_names:
        assert flux in output
        assert isinstance(output[flux], torch.Tensor)
    for state in model_state_names:
        assert state in output
        assert isinstance(output[state], torch.Tensor)


def test_hbv_parameter_coverage_negative():
    """当估计参数不覆盖物理参数时应触发断言（只测试构造）"""
    print("\n--- Running Test: HBV Coverage Negative ---")
    bad_cfg = {
        "hydrology_model": {
            "name": "hbv",
            "input_names": ["P", "Ep", "T"],
        },
        # 故意少给参数，触发model中的断言
        "static_estimator": {
            "name": "mlp",
            "estimate_parameters": ["TT", "CFMAX"],
            "input_names": ["attr1", "attr2"],
        },
        "warm_up": 0,
        "hru_num": 2,
    }
    with pytest.raises(AssertionError):
        _ = DplHydroModel(bad_cfg)