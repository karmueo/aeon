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

## 开发约束

1. 不得采用只解决局部问题的补丁式修改而忽视整体设计与全局优化
2. 不得引入过多用于中间通信的中间状态以免降低可读性并形成循环依赖
3. 不得为过渡场景编写大量防御性代码以免掩盖主逻辑并增加维护成本
4. 不得只追求功能完成而忽略架构设计
5. 不得省略必要注释，代码必须对他人和未来维护者可理解
6. 不得编写难以阅读的代码，必须保持结构简单清晰并添加解释性注释
7. 不得违反 SOLID 与 DRY 原则，必须保持职责单一并避免逻辑重复
8. 不得维护复杂的中间状态，仅允许保留最小必要的核心数据
9. 不得依赖外部或临时中间状态驱动 UI，所有 UI 状态必须从核心数据推导
10. 不得通过隐式或间接方式变更状态，状态变化应直接更新数据并由框架重新计算
11. 不得编写过量的防御性代码，应通过清晰的数据约束与边界设计解决问题
12. 不得保留未被使用的变量和函数
13. 不得将状态提升或集中到不必要的层级，状态应在最接近使用的位置管理
14. 不得在业务代码中直接依赖具体实现细节或硬编码外部服务
15. 不得在核心业务逻辑中混入 IO、网络、数据库等副作用操作
16. 不得形成隐式依赖，如依赖调用顺序、全局初始化或副作用时序
17. 不得吞掉异常或使用空 catch 掩盖错误
18. 不得将异常作为正常控制流的一部分
19. 不得返回语义不清或混用的错误结果（如 null / undefined / false）
20. 不得在多个位置同时维护同一份事实数据
21. 不得在未定义生命周期和失效策略的情况下缓存状态
22. 不得跨请求共享可变状态，除非明确设计为并发安全
23. 不得使用语义模糊或误导性的命名
24. 不得让单个函数或模块承担多个不相关语义
25. 不得引入非必要的时间耦合或隐含时间假设
26. 不得在关键路径中引入不可控的复杂度或隐式状态机
27. 不得臆测接口行为，必须先查询文档、定义或源码
28. 不得在需求、边界或输入输出不清晰的情况下直接实现
29. 不得基于猜测实现业务逻辑，必须与人类确认需求并留痕
30. 不得在未评估现有实现的情况下新增接口或模块
31. 不得跳过验证流程，必须编写并执行测试用例
32. 不得触碰架构红线或绕过既有设计规范
33. 不得假装理解需求或技术细节，不清楚时必须明确说明
34. 不得在缺乏上下文理解的情况下直接修改代码，必须基于整体结构审慎重构
