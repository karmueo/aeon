# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

aeon 是一个与 scikit-learn 兼容的时间序列机器学习工具包。它提供分类、回归、预测、聚类、异常检测、分割和相似性搜索等任务的算法。

## 常用命令

### 测试
```bash
# 运行所有测试（并行，使用逻辑核心数）
python -m pytest -n logical

# 运行特定文件的测试
python -m pytest aeon/classification/tests/test_specific.py

# 运行单个测试函数
python -m pytest aeon/classification/tests/test_specific.py::test_function_name

# 运行 doctest
python -m pytest -n logical --doctest-only --doctest-continue-on-failure

# 运行代码覆盖率测试
python -m pytest -n logical --cov=aeon --cov-report=xml

# 运行多线程估算器测试
python -m pytest aeon/testing/tests/ --enablethreading true -k "check_estimator_multithreading"
```

### 安装
```bash
# 开发安装（包含测试依赖）
pip install -e .[dev]

# 安装所有可选依赖
pip install -e .[all_extras]

# 仅安装深度学习依赖
pip install -e .[dl]
```

### 代码质量
```bash
# 运行 pre-commit 钩子（格式化和 lint）
pre-commit run --all-files

# 安装 pre-commit 钩子
pre-commit install
```

## 架构概览

### 核心设计模式

aeon 采用严格的**三层继承体系**，所有估算器都继承自 `aeon.base` 中的基类：

```
BaseAeonEstimator (继承自 sklearn.base.BaseEstimator)
    ├─ BaseSeriesEstimator     # 处理单个时间序列
    └─ BaseCollectionEstimator  # 处理时间序列集合（大多数分类/回归算法）
```

### 标签系统（Tag System）

标签系统是 aeon 的核心元数据机制，定义在 `aeon/utils/tags/` 中：

- **_tags**：类级别静态标签，声明估算器能力
- **_tags_dynamic**：实例级别的动态标签，可运行时修改

关键标签类别：
- `algorithm_type`：dictionary, distance, feature, interval, convolution, shapelet, deeplearning, hybrid
- `capability:multivariate`：是否支持多变量时间序列
- `capability:unequal_length`：是否支持不等长序列
- `capability:missing_values`：是否处理缺失值

### 数据类型抽象

aeon 定义了严格的时间序列数据类型规范（`aeon/utils/validation/`）：

**单个时间序列类型**：
- `pd.Series`、`pd.DataFrame`、`np.ndarray`

**时间序列集合类型**：
- `numpy3D`：形状为 (n_instances, n_channels, n_timepoints) 的 np.ndarray
- `np-list`：包含 2D np.ndarray 的列表（支持不等长）
- `df-list`：包含 pd.DataFrame 的列表（支持不等长）
- `pd-wide`：单索引 DataFrame，每行是一个时间序列
- `pd-multiindex`：多索引 DataFrame

### 模块组织

**任务模块**（按功能领域组织）：
- `aeon.classification` - 时间序列分类
  - `distance_based/` - 基于距离的分类器（如 KNN）
  - `interval_based/` - 基于区间的分类器
  - `dictionary_based/` - 基于字典的分类器
  - `feature_based/` - 基于特征的分类器
  - `shapelet_based/` - 基于形状子的分类器
  - `deep_learning/` - 深度学习分类器
- `aeon.regression` - 时间序列回归
- `aeon.forecasting` - 时间序列预测（实验性）
- `aeon.clustering` - 时间序列聚类
- `aeon.anomaly_detection` - 异常检测（实验性）
- `aeon.segmentation` - 时间序列分割（实验性）
- `aeon.similarity_search` - 相似性搜索（实验性）

**支持模块**：
- `aeon.transformations` - 数据转换（基于特征、区间、字典、卷积等）
- `aeon.distances` - 时间序列距离度量（DTW、Euclidean 等）
- `aeon.networks` - 深度学习网络架构（TensorFlow/Keras/PyTorch）
- `aeon.benchmarking` - 评估和基准测试工具
- `aeon.datasets` - 加载时间序列数据集

**基础设施模块**：
- `aeon.base` - 所有估算器的基础类
- `aeon.utils` - 验证、转换、标签管理、数据类型工具
- `aeon.testing` - 测试工具和辅助函数

### 组合模式

aeon 通过组合模式支持复杂的估算器构建：

- **Pipeline**：`aeon.pipeline` 串行组合转换和估算器
- **Ensemble**：集成多个估算器（如 `CollectionEnsemble`）
- **Composable**：通过 `ComposableEstimatorMixin` 支持任意组合

### 深度学习后端

aeon 支持多个深度学习框架：
- TensorFlow（>=2.14，Python <3.13）
- Keras（>=3.6.0，Python <3.13）
- PyTorch（>=1.13.1）

深度学习估算器位于 `aeon.networks/`，定义了可复用的网络架构。

## 实验性模块警告

以下模块不适用弃用政策，API 可能随时变化：
- `anomaly_detection`
- `forecasting`
- `segmentation`
- `similarity_search`
- `visualisation`
- `transformations.collection.self_supervised`
- `transformations.collection.imbalance`

## 重要约定

1. **继承层次**：新估算器必须继承自适当的基类（`BaseSeriesEstimator` 或 `BaseCollectionEstimator`）
2. **标签定义**：通过 `_tags` 类属性定义估算器能力
3. **数据验证**：使用 `aeon.utils.validation` 中的工具进行输入验证
4. **scikit-learn 兼容**：所有估算器必须实现 `fit`、`predict` 等 scikit-learn 标准 API
5. **numba 加速**：性能关键代码使用 `@numba.jit` 装饰器加速

## 类型注解

项目使用 Python `typing` 模块进行类型注解。运行 mypy 类型检查：
```bash
mypy aeon/
```
