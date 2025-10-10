import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hydlpy.model import DplHydroModel
from hydlpy.estimators.implements.mlp import MlpEstimator, MlpModel
from hydlpy.estimators.implements.lstm import LstmEstimator, LstmModel
from hydlpy.estimators.implements.direct import DirectEstimator, DirectModel


@pytest.fixture(scope="module")
def mlp_model_config():
    """MLP模型配置fixture"""
    return {
        "input_size": 10,
        "output_size": 20,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
    }


@pytest.fixture(scope="module")
def lstm_model_config():
    """LSTM模型配置fixture"""
    return {
        "input_size": 5,
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 10,
        "dropout": 0.1,
    }


@pytest.fixture(scope="module")
def direct_model_config():
    """Direct模型配置fixture"""
    return {"output_size": 10}


@pytest.fixture(scope="module")
def estimator_config():
    """Estimator配置fixture"""
    return {
        "hru_num": 8,
        "input_names": ["attr1", "attr2", "attr3"],
        "estimate_parameters": ["param1", "param2"],
    }


class TestMlpEstimator:
    """测试MLP Estimator"""

    def test_mlp_model_initialization(self, mlp_model_config):
        """测试MLP模型初始化"""
        print(f"\n--- Running Test: MLP Model Initialization ---")
        model = MlpModel(**mlp_model_config)

        assert isinstance(model.input_layer, nn.Linear)
        assert isinstance(model.fc_layer, nn.Linear)
        assert len(model.hidden_layer) == 2
        assert isinstance(model.dropout_layer, nn.Dropout)
        assert isinstance(model.sigmoid_layer, nn.Sigmoid)

    def test_mlp_model_forward(self):
        """测试MLP模型前向传播"""
        print(f"\n--- Running Test: MLP Model Forward ---")
        model = MlpModel(
            input_size=5, output_size=10, hidden_size=32, num_layers=1, dropout=0.0
        )

        batch_size = 20
        input_tensor = torch.rand((batch_size, 5))

        output = model(input_tensor)

        assert output.shape == (batch_size, 10)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # sigmoid输出范围

    def test_mlp_estimator_initialization(self, estimator_config):
        """测试MLP Estimator初始化"""
        print(f"\n--- Running Test: MLP Estimator Initialization ---")
        estimator = MlpEstimator(
            **estimator_config, hidden_size=64, num_layers=2, dropout=0.1
        )

        assert estimator.hru_num == 8
        assert estimator.input_size == 3
        assert estimator.output_size == (8, 2)
        assert estimator.estimate_parameters == ["param1", "param2"]

    def test_mlp_estimator_forward(self):
        """测试MLP Estimator前向传播"""
        print(f"\n--- Running Test: MLP Estimator Forward ---")
        estimator = MlpEstimator(
            hru_num=4,
            input_names=["attr1", "attr2", "attr3"],
            estimate_parameters=["param1", "param2"],
        )

        batch_size = 10
        input_tensor = torch.rand((batch_size, 3))

        output = estimator(input_tensor)

        assert "param1" in output
        assert "param2" in output
        assert output["param1"].shape == (batch_size, 4)
        assert output["param2"].shape == (batch_size, 4)


class TestLstmEstimator:
    """测试LSTM Estimator"""

    def test_lstm_model_initialization(self, lstm_model_config):
        """测试LSTM模型初始化"""
        print(f"\n--- Running Test: LSTM Model Initialization ---")
        model = LstmModel(**lstm_model_config)

        assert isinstance(model.input_layer, nn.Linear)
        assert isinstance(model.lstm_layer, nn.LSTM)
        assert isinstance(model.fc_layer, nn.Linear)
        assert isinstance(model.sigmoid_layer, nn.Sigmoid)

    def test_lstm_model_forward(self):
        """测试LSTM模型前向传播"""
        print(f"\n--- Running Test: LSTM Model Forward ---")
        model = LstmModel(
            input_size=3, hidden_size=32, num_layers=1, output_size=6, dropout=0.0
        )

        time_len, batch_size = 50, 20
        input_tensor = torch.rand((time_len, batch_size, 3))

        output = model(input_tensor)

        assert output.shape == (time_len, batch_size, 6)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # sigmoid输出范围

    def test_lstm_estimator_initialization(self, estimator_config):
        """测试LSTM Estimator初始化"""
        print(f"\n--- Running Test: LSTM Estimator Initialization ---")
        estimator = LstmEstimator(
            **estimator_config, hidden_size=64, num_layers=2, dropout=0.1
        )

        assert estimator.hru_num == 8
        assert estimator.input_size == 3
        assert estimator.output_size == (8, 2)
        assert estimator.estimate_parameters == ["param1", "param2"]

    def test_lstm_estimator_forward(self):
        """测试LSTM Estimator前向传播"""
        print(f"\n--- Running Test: LSTM Estimator Forward ---")
        estimator = LstmEstimator(
            hru_num=4,
            input_names=["prcp", "pet", "temp"],
            estimate_parameters=["param1", "param2"],
        )

        time_len, batch_size = 100, 10
        input_tensor = torch.rand((time_len, batch_size, 3))

        output = estimator(input_tensor)

        assert "param1" in output
        assert "param2" in output
        assert output["param1"].shape == (time_len, batch_size, 4)
        assert output["param2"].shape == (time_len, batch_size, 4)


class TestDirectEstimator:
    """测试Direct Estimator"""

    def test_direct_model_initialization(self, direct_model_config):
        """测试Direct模型初始化"""
        print(f"\n--- Running Test: Direct Model Initialization ---")
        model = DirectModel(**direct_model_config)

        assert isinstance(model.model_params, nn.Parameter), f"模型参数类型需要为`nn.Parameter`"
        assert model.model_params.shape == (10,), f"模型参数类型的维度需要为`(10,)`"
        assert isinstance(model.sigmoid_layer, nn.Sigmoid)

    def test_direct_model_forward(self):
        """测试Direct模型前向传播"""
        print(f"\n--- Running Test: Direct Model Forward ---")
        model = DirectModel(output_size=5)

        batch_size = 20
        input_tensor = torch.rand((batch_size, 3))  # 输入被忽略

        output = model(input_tensor)

        assert output.shape == (batch_size, 5)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # sigmoid输出范围

    def test_direct_estimator_initialization(self, estimator_config):
        """测试Direct Estimator初始化"""
        print(f"\n--- Running Test: Direct Estimator Initialization ---")
        estimator = DirectEstimator(**estimator_config)

        assert estimator.hru_num == 8
        assert estimator.input_size == 3
        assert estimator.output_size == (8, 2)
        assert estimator.estimate_parameters == ["param1", "param2"]

    def test_direct_estimator_forward(self):
        """测试Direct Estimator前向传播"""
        print(f"\n--- Running Test: Direct Estimator Forward ---")
        estimator = DirectEstimator(
            hru_num=4,
            input_names=["attr1", "attr2"],
            estimate_parameters=["param1", "param2"],
        )

        batch_size = 10
        input_tensor = torch.rand((batch_size, 2))

        output = estimator(input_tensor)

        assert "param1" in output
        assert "param2" in output
        assert output["param1"].shape == (batch_size, 4)
        assert output["param2"].shape == (batch_size, 4)



class TestEstimatorIntegration:
    """测试Estimator与DplHydroModel的集成"""

    @pytest.mark.parametrize(
        "estimator_name,estimator_config",
        [
            (
                "mlp",
                {
                    "name": "mlp",
                    "estimate_parameters": ["Qmax", "f", "Tmin", "Tmax", "Df", "Smax"],
                    "input_names": ["attr1", "attr2", "attr3", "attr4"],
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.1,
                },
            ),
            (
                "direct",
                {
                    "name": "direct",
                    "estimate_parameters": ["Qmax", "f", "Tmin", "Tmax", "Df", "Smax"],
                    "input_names": ["attr1", "attr2"],
                },
            ),
        ],
    )
    def test_estimator_integration(self, estimator_name, estimator_config):
        """测试Estimator集成"""
        print(f"\n--- Running Test: {estimator_name.upper()} Estimator Integration ---")

        config = {
            "hydrology_model": {
                "name": "exphydro",
                "input_names": ["prcp", "pet", "temp"],
            },
            "static_estimator": estimator_config,
            "warm_up": 100,
            "hru_num": 8,
        }

        model = DplHydroModel(config)

        time_len, basin_num = 200, 20
        input_data = {
            "x_phy": torch.rand((time_len, basin_num, 3)),
            "x_nn_norm": torch.rand((time_len, basin_num, 3)),
            "xc_nn_norm": torch.rand((time_len, basin_num, 3)),
            "c_nn_norm": torch.rand((basin_num, len(estimator_config["input_names"]))),
        }

        output = model(input_data)
        assert "flow" in output

    def test_lstm_estimator_integration(self):
        """测试LSTM Estimator集成"""
        print(f"\n--- Running Test: LSTM Estimator Integration ---")

        config = {
            "hydrology_model": {
                "name": "exphydro",
                "input_names": ["prcp", "pet", "temp"],
            },
            "dynamic_estimator": {
                "name": "lstm",
                "estimate_parameters": ["Qmax", "f", "Tmin", "Tmax", "Df", "Smax"],
                "input_names": ["prcp", "pet", "temp"],
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.1,
            },
            "warm_up": 100,
            "hru_num": 8,
        }

        model = DplHydroModel(config)

        time_len, basin_num = 200, 20
        input_data = {
            "x_phy": torch.rand((time_len, basin_num, 3)),
            "x_nn_norm": torch.rand((time_len, basin_num, 3)),
            "xc_nn_norm": torch.rand((time_len, basin_num, 3)),
            "c_nn_norm": torch.rand((basin_num, 3)),
        }

        output = model(input_data)
        assert "flow" in output

    def test_combined_estimators(self):
        """测试组合estimator"""
        print(f"\n--- Running Test: Combined Estimators ---")

        config = {
            "hydrology_model": {
                "name": "exphydro",
                "input_names": ["prcp", "pet", "temp"],
            },
            "static_estimator": {
                "name": "mlp",
                "estimate_parameters": ["Qmax", "f", "Tmin", "Tmax", "Df", "Smax"],
                "input_names": ["attr1", "attr2", "attr3", "attr4"],
            },
            "dynamic_estimator": {
                "name": "lstm",
                "estimate_parameters": ["Qmax", "f"],
                "input_names": ["prcp", "pet", "temp"],
            },
            "warm_up": 100,
            "hru_num": 8,
        }

        model = DplHydroModel(config)

        time_len, basin_num = 200, 20
        input_data = {
            "x_phy": torch.rand((time_len, basin_num, 3)),
            "x_nn_norm": torch.rand((time_len, basin_num, 3)),
            "xc_nn_norm": torch.rand((time_len, basin_num, 3)),
            "c_nn_norm": torch.rand((basin_num, 4)),
        }

        output = model(input_data)
        assert "flow" in output


class TestEstimatorEdgeCases:
    """测试Estimator边界情况"""

    @pytest.mark.parametrize("hru_num", [1, 4, 8, 16])
    def test_single_hru(self, hru_num):
        """测试单个HRU"""
        print(f"\n--- Running Test: Single HRU {hru_num} ---")

        estimator = MlpEstimator(
            hru_num=hru_num,
            input_names=["attr1"],
            estimate_parameters=["param1"],
        )

        batch_size = 5
        input_tensor = torch.rand((batch_size, 1))

        output = estimator(input_tensor)

        assert "param1" in output
        assert output["param1"].shape == (batch_size, hru_num)

    def test_single_parameter(self):
        """测试单个参数"""
        print(f"\n--- Running Test: Single Parameter ---")

        estimator = MlpEstimator(
            hru_num=4,
            input_names=["attr1", "attr2", "attr3"],
            estimate_parameters=["param1"],
        )

        batch_size = 10
        input_tensor = torch.rand((batch_size, 3))

        output = estimator(input_tensor)

        assert "param1" in output
        assert output["param1"].shape == (batch_size, 4)

    def test_large_batch_size(self):
        """测试大批次大小"""
        print(f"\n--- Running Test: Large Batch Size ---")

        estimator = LstmEstimator(
            hru_num=8,
            input_names=["prcp", "pet", "temp"],
            estimate_parameters=["param1", "param2"],
        )

        time_len, batch_size = 1000, 100
        input_tensor = torch.rand((time_len, batch_size, 3))

        output = estimator(input_tensor)

        assert "param1" in output
        assert "param2" in output
        assert output["param1"].shape == (time_len, batch_size, 8)
        assert output["param2"].shape == (time_len, batch_size, 8)


@pytest.mark.parametrize(
    "model_type,model_class",
    [
        ("MlpModel", MlpModel),
        ("LstmModel", LstmModel),
        ("DirectModel", DirectModel),
    ],
)
class TestModelClasses:
    """测试模型类"""

    def test_model_class_instantiation(self, model_type, model_class):
        """测试模型类实例化"""
        print(f"\n--- Running Test: {model_type} Class Instantiation ---")

        if model_type == "MlpModel":
            model = model_class(
                input_size=5, output_size=10, hidden_size=32, num_layers=1, dropout=0.0
            )
        elif model_type == "LstmModel":
            model = model_class(
                input_size=3, hidden_size=32, num_layers=1, output_size=6, dropout=0.0
            )
        elif model_type == "DirectModel":
            model = model_class(output_size=5)

        assert isinstance(model, model_class)
        assert hasattr(model, "forward")

    def test_model_forward_method(self, model_type, model_class):
        """测试模型前向传播方法"""
        print(f"\n--- Running Test: {model_type} Forward Method ---")

        if model_type == "MlpModel":
            model = model_class(
                input_size=5, output_size=10, hidden_size=32, num_layers=1, dropout=0.0
            )
            input_tensor = torch.rand((10, 5))
        elif model_type == "LstmModel":
            model = model_class(
                input_size=3, hidden_size=32, num_layers=1, output_size=6, dropout=0.0
            )
            input_tensor = torch.rand((50, 10, 3))
        elif model_type == "DirectModel":
            model = model_class(output_size=5)
            input_tensor = torch.rand((10, 3))

        output = model(input_tensor)
        assert isinstance(output, torch.Tensor)
        assert output.dim() >= 1
