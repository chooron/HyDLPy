# examples/run_advanced_model.py
import torch
import sympy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 假设你的包已经正确安装
from dlhpy.hydrology_cores.base import HydrologyModel, hydroflux, stateflux

# 1. 定义一个具有内部依赖关系的水文模型
class AdvancedBucketModel(HydrologyModel):
    """
    一个更高级的水桶模型，包含依赖关系以测试拓扑排序。
    - 状态(S)的变化 = 降雨(P) - 实际蒸散发(AET) - 产流(runoff)
    - AET 依赖于 潜在蒸散发(PET) 和当前状态(S)
    - runoff 依赖于当前状态(S)
    执行顺序应该是 AET 和 runoff (任意顺序)，然后是 S。
    """
    # 1. 只需“注册”状态和参数
    _state_variables = ["S"]
    _parameter_bounds = {"k": (0.01, 1.0), "c": (0.0, 0.5)}

    # 2. 直接开始定义公式，使用单一装饰器
    @hydroflux
    def runoff(k, S):
        return k * S

    @hydroflux
    def actual_evaporation(PET, S, c):
        return PET * sympy.Min(1.0, S * c)

    @stateflux
    def S(P, runoff, actual_evaporation):
        return P - runoff - actual_evaporation

if __name__ == "__main__":
    print("--- Running DLHPy Test Case ---")

    # 2. 准备模型输入数据
    N_TIMESTEPS = 365  # 模拟一年的日尺度数据
    N_BASINS = 10  # 同时模拟 10 个流域

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 创建模拟的强迫数据 (时间序列 x 流域数)
    x_dict = {
        "P": torch.rand(N_TIMESTEPS, N_BASINS, device=device) * 10,  # 日降雨量 0-10 mm
        "PET": torch.rand(N_TIMESTEPS, N_BASINS, device=device) * 5,  # 日潜蒸发 0-5 mm
    }

    # 3. 实例化模型
    # 我们可以在这里传入一些配置，例如 warmup
    model = AdvancedBucketModel(warm_up=30).to(device)

    # # 4. 准备模拟的、归一化后的模型参数 (流域数 x 参数数)
    # # 注意：参数数量和顺序是从模型属性中自动获取的
    # num_params = len(model._parameter_variables)
    # # 假设参数是从某个神经网络输出的，这里用随机数模拟
    # params_norm = torch.randn(N_BASINS, num_params, device=device)

    # # 5. 运行模型
    # print("\nCalling model.forward()...")
    # outputs = model(x_dict=x_dict, params_norm=params_norm)
    # print("Model execution finished.\n")

    # # 6. 检查和打印输出
    # print("--- Outputs ---")
    # for name, tensor in outputs.items():
    #     print(f"Output '{name}' shape: {tensor.shape}")

    # # 验证输出的时间步长是否正确（总时长 - 预热期）
    # expected_timesteps = N_TIMESTEPS - model.warm_up
    # assert outputs['S'].shape[0] == expected_timesteps
    # print(f"\nAssertion passed: Output timesteps ({outputs['S'].shape[0]}) match expected ({expected_timesteps}).")
