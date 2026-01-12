# Repository Guidelines

## 项目结构与模块组织

- 核心源码在 `aeon/`，按任务领域拆分（如 `classification/`、`regression/`、`forecasting/`、`transformations/`）。
- 单元测试通常位于对应包内的 `tests/` 目录（如 `aeon/segmentation/tests/`）。
- 文档在 `docs/`，示例与教程在 `examples/`；项目级配置位于 `pyproject.toml` 与 `conftest.py`。
- 项目使用conda虚拟环境`time-series`。

## 构建、测试与本地开发命令

- `pip install --editable .[dev]`：安装开发依赖并启用可编辑模式。
- `pip install --editable .[dev,all_extras]`：包含可选依赖，用于扩展功能或更完整的测试。
- `pre-commit install`：安装提交前检查（PR 必需）。
- `pytest aeon/`：运行全部单元测试。
- `pytest aeon/ -k DummyClassifier`：按关键字筛选测试。
- `pytest aeon/ -n auto`：并行测试（需 `pytest-xdist`）。
- 额外参数：`--nonumba`（禁用 numba 编译）、`--enablethreading`、`--prtesting`。

## 编码风格与命名规范

- 遵循 PEP8；`black` 默认格式化，`flake8` 行宽 88。
- 使用 `isort` 排序导入，`ruff` 的 `pydocstyle` 强制 `numpydoc` 风格文档。
- 变量/函数用下划线分词（如 `n_cases`），类名用驼峰；允许 `X`/`X_train` 表示数据。
- 使用绝对导入；禁止 `import *`。

## 测试指南

- 测试文件名以 `test_` 开头，函数名以 `test_` 开头。
- 测试放在与模块同级的 `tests/` 目录（如 `module.py` 对应 `tests/test_module.py`）。
- 需要软依赖的测试请用 `pytest.mark.skipif` 处理。

## 提交与 PR 指南

- Git 历史中常见提交前缀：`[BUG]`、`[ENH]`、`[DOC]`、`[MNT]` 等，建议在提交消息中使用并带 PR 编号（如 `(#1234)`）。
- PR 必须遵循模板，勿删除模板内容；标题需带合适的标签（见 `docs/contributing/issues.md`）。
- 使用他人实现或受其启发时需在 PR 中注明来源与许可证。
- 尽量避免强制推送；建议在测试通过后再请求评审。

## 依赖与配置提示

- 本项目使用 `pre-commit` 强制格式化与静态检查；未通过检查的 PR 通常不予合并。
- 覆盖率由 `pytest-cov`/`coverage`/`codecov` 跟踪，CI 可能对 `numba` 相关代码有特殊设置。