#!/usr/bin/env python3
"""
测试新的 @hydroflux 和 @stateflux 装饰器功能
"""

import sympy
from dlhpy.hydrology_cores.base import HydrologyModel, hydroflux, stateflux

class TestModel(HydrologyModel):
    """测试模型，验证装饰器功能"""
    
    _parameter_bounds = {"k": (0.01, 1.0), "c": (0.0, 0.5)}
    
    # 定义符号
    S, P, PET, k, c, runoff, evap = sympy.symbols("S P PET k c runoff evap")
    
    # 测试 @hydroflux 装饰器
    @hydroflux
    def evap(S, PET, c):
        """实际蒸散发通量"""
        return PET * sympy.Min(1.0, S * c)
    
    # 测试 @stateflux 装饰器
    @stateflux
    def S(P, runoff, evap):
        """状态变量变化率"""
        return P - runoff - evap
    
    # 普通静态方法（不需要特殊标记）
    @staticmethod
    def runoff(k, S):
        """产流通量"""
        return k * S

def test_decorator_functionality():
    """测试装饰器功能"""
    print("=== 测试装饰器功能 ===")
    
    # 测试 @hydroflux 装饰器
    print(f"evap 方法类型: {type(TestModel.evap)}")
    print(f"evap 是静态方法: {isinstance(TestModel.evap, staticmethod)}")
    print(f"evap 有 _is_hydroflux 标记: {hasattr(TestModel.evap.__func__, '_is_hydroflux')}")
    
    # 测试 @stateflux 装饰器
    print(f"S 方法类型: {type(TestModel.S)}")
    print(f"S 是静态方法: {isinstance(TestModel.S, staticmethod)}")
    print(f"S 有 _is_stateflux 标记: {hasattr(TestModel.S.__func__, '_is_stateflux')}")
    
    # 测试普通静态方法
    print(f"runoff 方法类型: {type(TestModel.runoff)}")
    print(f"runoff 是静态方法: {isinstance(TestModel.runoff, staticmethod)}")
    print(f"runoff 有 _is_hydroflux 标记: {hasattr(TestModel.runoff, '_is_hydroflux')}")
    
    # 测试方法调用
    print("\n=== 测试方法调用 ===")
    
    # 测试 evap 方法调用
    result_evap = TestModel.evap(10.0, 5.0, 0.3)  # S=10, PET=5, c=0.3
    print(f"evap(10, 5, 0.3) = {result_evap}")
    
    # 测试 S 方法调用
    result_S = TestModel.S(20.0, 3.0, 2.0)  # P=20, runoff=3, evap=2
    print(f"S(20, 3, 2) = {result_S}")
    
    # 测试 runoff 方法调用
    result_runoff = TestModel.runoff(0.5, 10.0)  # k=0.5, S=10
    print(f"runoff(0.5, 10) = {result_runoff}")

if __name__ == "__main__":
    test_decorator_functionality()
