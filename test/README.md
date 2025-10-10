# DplHydroModel 测试套件

这个目录包含了DplHydroModel的完整测试套件，用于验证模型的各种功能和性能。

## 测试文件说明

### 1. `test_dpl_model_simple.py`

- **用途**: 基础功能测试
- **内容**:
  - 模型初始化测试
  - 前向传播测试
  - 无estimator模型测试
  - 不同estimator测试
  - 训练功能测试
- **运行时间**: ~30秒
- **适用场景**: 快速验证基本功能

### 2. `test_dpl_model_comprehensive.py`

- **用途**: 综合功能测试
- **内容**:
  - 基础功能测试
  - 不同estimator组合测试
  - 不同水文模型测试
  - 边界情况测试
  - 训练相关功能测试
  - 参数估计功能测试
  - 错误处理测试
  - 别名映射功能测试
- **运行时间**: ~2-3分钟
- **适用场景**: 完整功能验证

### 3. `test_estimators.py`

- **用途**: Estimator组件测试
- **内容**:
  - MLP Estimator测试
  - LSTM Estimator测试
  - Direct Estimator测试
  - Estimator集成测试
  - 边界情况测试
- **运行时间**: ~1-2分钟
- **适用场景**: 专门测试参数估计器

### 4. `test_hydrology_models.py`

- **用途**: 水文模型测试
- **内容**:
  - ExpHydro模型测试
  - HBV模型测试
  - XAJ模型测试
  - 模型比较测试
- **运行时间**: ~1分钟
- **适用场景**: 专门测试水文模型

### 5. `test_performance.py`

- **用途**: 性能和可扩展性测试
- **内容**:
  - 前向传播时间测试
  - 内存使用测试
  - 批处理效率测试
  - 不同模型复杂度测试
  - 梯度计算性能测试
  - 可扩展性测试
- **运行时间**: ~3-5分钟
- **适用场景**: 性能基准测试

## 运行测试

### 方法1: 使用测试运行脚本（推荐）

```bash
# 运行所有测试
python test/run_all_tests.py

# 只运行简单测试
python test/run_all_tests.py --simple

# 只运行性能测试
python test/run_all_tests.py --performance

# 只运行estimator测试
python test/run_all_tests.py --estimators

# 只运行水文模型测试
python test/run_all_tests.py --hydrology

# 只运行综合测试
python test/run_all_tests.py --comprehensive

# 使用pytest运行测试
python test/run_all_tests.py --pytest

# 静默模式
python test/run_all_tests.py --quiet
```

### 方法2: 直接运行测试文件

```bash
# 运行简单测试
python test/test_dpl_model_simple.py

# 运行综合测试
python test/test_dpl_model_comprehensive.py

# 运行estimator测试
python test/test_estimators.py

# 运行水文模型测试
python test/test_hydrology_models.py

# 运行性能测试
python test/test_performance.py
```

### 方法3: 使用pytest

```bash
# 运行所有pytest测试
python -m pytest test/test_dpl_model_comprehensive.py -v

# 运行特定测试类
python -m pytest test/test_dpl_model_comprehensive.py::TestDplHydroModelBasic -v

# 运行特定测试方法
python -m pytest test/test_dpl_model_comprehensive.py::TestDplHydroModelBasic::test_model_initialization -v
```

## 测试覆盖范围

### 功能测试
- ✅ 模型初始化
- ✅ 前向传播
- ✅ 静态参数估计
- ✅ 动态参数估计
- ✅ 不同estimator组合
- ✅ 不同水文模型
- ✅ 训练功能
- ✅ 损失计算
- ✅ 优化器配置
- ✅ 别名映射

### 边界情况测试
- ✅ 最小配置
- ✅ 大量HRU
- ✅ 零预热期
- ✅ 单个HRU
- ✅ 单个参数
- ✅ 大批次大小
- ✅ 长时间序列

### 错误处理测试
- ✅ 无效水文模型名称
- ✅ 无效estimator名称
- ✅ 缺少必需配置
- ✅ 错误输入形状

### 性能测试
- ✅ 前向传播时间
- ✅ 内存使用
- ✅ 批处理效率
- ✅ 梯度计算性能
- ✅ 可扩展性

## 测试数据

测试使用随机生成的模拟数据，包括：
- 气象数据（降水、蒸散发、温度）
- 流域属性数据
- 目标值（用于训练测试）

## 注意事项

1. **内存要求**: 性能测试可能需要较多内存，建议至少4GB可用内存
2. **运行时间**: 完整测试套件需要5-10分钟
3. **依赖**: 确保已安装所有必需的依赖包
4. **GPU**: 测试默认使用CPU，如需GPU测试请修改代码

## 故障排除

### 常见问题

1. **ImportError**: 确保在项目根目录运行测试
2. **CUDA错误**: 检查PyTorch CUDA安装
3. **内存不足**: 减少测试数据大小或使用更小的模型配置
4. **测试超时**: 检查系统性能，可能需要调整超时设置

### 调试模式

```bash
# 启用详细输出
python test/run_all_tests.py --simple

# 使用pytest调试
python -m pytest test/test_dpl_model_simple.py -v -s
```

## 贡献

添加新测试时，请遵循以下原则：
1. 测试应该独立且可重复
2. 使用描述性的测试名称
3. 包含适当的断言
4. 添加必要的文档说明
5. 考虑边界情况和错误处理
