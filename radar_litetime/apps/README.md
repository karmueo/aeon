# UDP LITETime 雷达航迹分类应用

基于 aeon LITETime 模型的 UDP 组播航迹实时分类应用。

## 功能

1. **UDP 组播接收**：接收雷达航迹 UDP 组播报文
2. **报文解析**：解析二进制航迹报文
3. **特征提取与预处理**：提取特征并归一化
4. **LITETime 模型推理**：使用训练好的 LITETime 模型进行分类
5. **结果发布**：通过 UDP 组播发布分类结果

## 架构设计

采用模块化、松耦合的设计模式：

```
apps/
├── config/
│   ├── __init__.py
│   ├── loader.py          # 配置加载模块
│   └── default.yaml       # 默认配置
├── core/
│   ├── __init__.py
│   ├── predictor.py       # 预测器抽象与LITETime适配器
│   └── buffer.py          # 轨迹缓冲管理
├── utils/
│   ├── __init__.py
│   └── feature_processor.py  # 特征提取与归一化
└── udp_litetime_predictor.py # 主应用入口
```

### 设计模式

- **策略模式**：`BasePredictor` 抽象预测器接口，`LITETimePredictor` 实现
- **适配器模式**：将 aeon `LITETimeClassifier` 适配到统一接口
- **工厂模式**：`PredictorFactory` 负责创建预测器实例
- **单一职责**：每个模块职责清晰，便于维护和扩展

## 依赖

- Python >= 3.10
- aeon (已安装)
- numpy
- pyyaml

安装依赖：
```bash
pip install pyyaml
```

## 使用方法

### 1. 使用默认配置文件运行

```bash
cd /home/tl/work/aeon
python -m radar_litetime.apps.udp_litetime_predictor --config radar_litetime/apps/config/default.yaml
```

### 2. 使用命令行参数运行

```bash
# 基本运行
python -m radar_litetime.apps.udp_litetime_predictor \
    --in_group 230.1.1.22 \
    --in_port 8002 \
    --out_group 230.1.1.24 \
    --out_port 8011 \
    --meta_file radar_litetime/output/meta.json

# 启用调试输出
python -m radar_litetime.apps.udp_litetime_predictor \
    --config radar_litetime/apps/config/default.yaml \
    --print_targets \
    --print_features
```

### 3. 本地文件测试模式

```bash
python -m radar_litetime.apps.udp_litetime_predictor \
    --local_test \
    --local_test_path mydataset/radar_augv3/bird/sample.xls \
    --local_test_points 20 \
    --meta_file radar_litetime/output/meta.json
```

## 配置说明

配置文件采用 YAML 格式，支持命令行参数覆盖：

### 接收器配置 (receiver)
- `group`: 输入组播地址
- `port`: 输入端口
- `iface`: 输入网卡接口
- `timeout_s`: 接收超时时间

### 发布器配置 (publisher)
- `group`: 输出组播地址
- `port`: 输出端口
- `iface`: 输出网卡接口
- `ttl`: 组播 TTL

### 预测器配置 (predictor)
- `model_dir`: 模型目录路径
- `meta_file`: meta.json 文件路径

### 缓冲区配置 (buffer)
- `seq_len`: 序列窗口长度
- `min_seq_len`: 最小序列长度
- `max_age_s`: 轨迹最大保留时间
- `window_step`: 滑窗步长 (0=每条都推理)

### 推理配置 (inference)
- `publish_interval_ms`: 发布节流毫秒 (0=不节流)
- `use_batch_ema`: 同批号目标使用 EMA 平滑
- `ema_alpha`: EMA 平滑系数

## 复用的模块

从 `Time-Series-Library` 复用以下模块：
- `udp.receiver.MulticastReceiver` - UDP 组播接收
- `udp.parser` - UDP 报文解析
- `udp.publisher.MulticastPublisher` - UDP 组播发送

## 扩展性

如需支持其他模型，只需继承 `BasePredictor` 并实现相应接口：

```python
class MyPredictor(BasePredictor):
    def load(self, num_features: int) -> None:
        # 加载模型
        pass

    def predict(self, batch: np.ndarray, lengths: np.ndarray) -> tuple:
        # 执行预测
        pass
```
