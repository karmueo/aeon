# 雷达 MultiROCKET 分类（鸟 vs 无人机）

本目录提供一个最小可用的训练/评估流程，用于识别 Excel 轨迹数据中的鸟与无人机。
使用 `aeon` 的 `MultiRocketClassifier`，数据目录结构如下：

```
mydataset/radar_augv3/
  bird/
    *.xls
  uav/
    *.xls
```

每个 `.xls` 文件视为一个样本。数值列会作为通道（非数值列自动忽略），每一行代表一个时间步。
如果文件长度不一致，可通过参数进行补齐或截断到固定长度。

## 依赖与安装

- Python 依赖：`aeon`, `scikit-learn`, `joblib`, `pandas`, `xlrd`
- 示例安装（仓库内执行）：

```
pip install --editable .[dev]
pip install pandas xlrd
```

## 训练

```
python radar_multirocket/train.py \
  --data-dir mydataset/radar_augv3 \
  --output-dir radar_multirocket/output \
  --classes bird,uav \
  --extensions .xls \
  --pad-mode pad \
  --pad-value 0 \
  --nan-strategy zero
```

输出文件：
- `radar_multirocket/output/model.joblib`
- `radar_multirocket/output/split.json`
- `radar_multirocket/output/meta.json`

## 测试

```
python radar_multirocket/test.py \
  --data-dir mydataset/radar_augv3 \
  --model-path radar_multirocket/output/model.joblib \
  --split-file radar_multirocket/output/split.json
```

## 说明

- `.xls` 中存在多列数值时，会被视为多通道输入。
- 若要求严格等长输入，训练时设置 `--pad-mode error`。
- 若需固定长度，训练时设置 `--max-length`，测试会从 `meta.json` 复用该长度。

## 参数调优建议（针对当前数据）

已知条件：每条航迹报文长度约 20 个点，默认使用 14 个特征（通道），单类约 13000 条报文。

- `--n-kernels`（核数量）：主要影响特征维度与训练时间。样本量较多时可提高到 10000~20000；若训练过慢或内存不足，可先降到 2000~5000 做验证。
- `--n-features-per-kernel`（每核特征数）：MultiROCKET 默认 4，属于稳妥起点。你的序列较短（20 点）且通道数 14，通常保持 4 即可；如果需要更强表达力可尝试 6~8，但训练时间与内存会增加。
- `--max-dilations-per-kernel`（每核最大膨胀数）：控制不同尺度的感受野数量。序列较短时，过大的膨胀数收益有限且会增加特征计算量。建议从默认 32 起步；若训练太慢或收益不明显，可降到 16；若需要更强的多尺度表达且资源允许，可尝试 48 或 64。

推荐起步组合：`--n-kernels 10000 --n-features-per-kernel 4`。如需更快迭代，先用 `--n-kernels 3000` 做小规模对比，再逐步增大。

## UDP 组播实时分类应用

`apps/udp_multirocket_predictor.py` 提供了 UDP 组播实时分类功能，与 LITETime 版本类似，但使用 MultiRocket 模型进行推理。

### 功能特性

- 接收 UDP 组播航迹报文，解析目标数据
- 使用 MultiRocket 模型进行实时分类
- 发布分类结果到组播组
- 支持多批号多航迹并发处理
- 支持滑动窗口推理和 EMA 平滑
- 支持本地文件测试模式

### 使用方法

#### 1. 使用配置文件（推荐）

编辑 `apps/config/default.yaml` 配置参数：

```bash
python radar_multirocket/apps/udp_multirocket_predictor.py \
  --config radar_multirocket/apps/config/default.yaml
```

#### 2. 使用命令行参数

```bash
python radar_multirocket/apps/udp_multirocket_predictor.py \
  --in_group 230.1.1.22 \
  --in_port 8002 \
  --in_iface 192.168.1.112 \
  --out_group 230.1.1.24 \
  --out_port 8011 \
  --out_iface 192.168.1.112 \
  --meta_file radar_multirocket/output/meta.json \
  --seq_len 20 \
  --min_seq_len 20
```

#### 3. 本地文件测试

使用本地 .xls/.csv 文件进行测试：

```bash
python radar_multirocket/apps/udp_multirocket_predictor.py \
  --local_test \
  --local_test_path test.csv \
  --local_test_points 20 \
  --meta_file radar_multirocket/output/meta.json
```

### 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | YAML 配置文件路径 | - |
| `--in_group` | 输入组播地址 | 230.1.1.22 |
| `--in_port` | 输入端口 | 8002 |
| `--out_group` | 输出组播地址 | 230.1.1.24 |
| `--out_port` | 输出端口 | 8011 |
| `--meta_file` | Meta JSON 文件路径 | output/meta.json |
| `--seq_len` | 序列长度 | 20 |
| `--min_seq_len` | 最小序列长度 | 20 |
| `--max_age_s` | 轨迹最大保留时间（秒） | 10.0 |
| `--window_step` | 滑窗步长（0=每条都推理） | 0 |
| `--publish_interval_ms` | 发布节流毫秒（0=不节流） | 0 |
| `--use_batch_ema` | 启用 EMA 平滑 | True |
| `--ema_alpha` | EMA 平滑系数 | 0.4 |

### 输出格式

应用会输出 JSON 格式的分类结果：

```json
{
  "batch_id": 123,
  "count": 1,
  "inference_time_ms": 5.23,
  "items": [
    {
      "batch_id": 123,
      "track_id": 1,
      "timestamp_ms": 1234567890,
      "time_25us": 49382715,
      "pred": 1,
      "prob_uav": 0.85,
      "prob_bird": 0.15
    }
  ]
}
```

### 与 LITETime 版本的差异

1. **模型加载方式**：MultiRocket 使用 `joblib.load()`，而 LITETime 使用深度学习模型加载
2. **无归一化配置**：MultiRocket 训练时已包含标准化（StandardScaler），推理时无需额外归一化
3. **其他功能**：完全复用了 UDP 组播、缓冲管理、EMA 平滑等功能

### 依赖项

UDP 组播应用需要 Time-Series-Library 项目中的 UDP 模块：

```bash
# 确保 Time-Series-Library 在项目根目录
Time-Series-Library/
  udp/
    parser.py
    publisher.py
    receiver.py
```

如果缺少依赖，请参考 `radar_litetime/README.md` 中的说明进行配置。
