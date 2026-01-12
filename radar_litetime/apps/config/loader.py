"""配置加载与管理模块。

提供从 YAML 配置文件和命令行参数加载配置的功能。
"""

import argparse
from pathlib import Path
from typing import Any


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """加载 YAML 配置文件。

    Args:
        config_path: YAML 配置文件路径。

    Returns:
        解析后的配置字典。
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "yaml 模块未安装，请运行: pip install pyyaml"
        ) from exc

    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config_with_args(
    config: dict[str, Any], args: argparse.Namespace
) -> argparse.Namespace:
    """将配置文件的值合并到参数中（仅在命令行未显式指定时）。

    Args:
        config: 从 YAML 加载的配置字典。
        args: 解析后的命令行参数。

    Returns:
        合并后的命名空间对象。
    """
    import sys

    # 记录命令行显式指定的参数
    cmd_line_args = set()
    for argv in sys.argv[1:]:
        if argv.startswith("--"):
            arg_name = argv[2:].split("=")[0]
            cmd_line_args.add(arg_name)

    # 配置文件字段到命令行参数的映射
    config_mapping = {
        "receiver": {
            "group": "in_group",
            "port": "in_port",
            "iface": "in_iface",
            "timeout_s": "timeout",
        },
        "publisher": {
            "group": "out_group",
            "port": "out_port",
            "iface": "out_iface",
        },
        "local_test": {
            "enabled": "local_test",
            "path": "local_test_path",
            "points": "local_test_points",
        },
        # 其他字段直接映射
        "buffer": None,
        "normalizer": None,
        "predictor": None,
        "inference": None,
    }

    for section, values in config.items():
        if not isinstance(values, dict):
            continue

        mapping = config_mapping.get(section, {})
        for key, value in values.items():
            # 确定目标参数名
            if mapping and key in mapping:
                arg_name = mapping[key]
            elif mapping is None:
                arg_name = key
            else:
                continue

            # 只有当命令行未显式指定时才使用配置文件的值
            if arg_name not in cmd_line_args and hasattr(args, arg_name):
                # 特殊处理 boolean 类型的 flag
                if key == "enabled" and isinstance(value, bool):
                    if value:
                        setattr(args, arg_name, True)
                else:
                    setattr(args, arg_name, value)

    return args


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。

    Returns:
        配置好的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(
        description="组播航迹接收 + LITETime 预测 + 组播发布",
        epilog="配置文件中的参数会被命令行参数覆盖。使用 --config 指定 YAML 配置文件。",
    )
    parser.add_argument("--config", default="", help="YAML 配置文件路径")

    # 接收器配置
    parser.add_argument("--in_group", default="", help="输入组播地址")
    parser.add_argument("--in_port", type=int, default=0, help="输入端口")
    parser.add_argument("--in_iface", default="0.0.0.0", help="输入网卡IP")
    parser.add_argument("--bind_ip", default="", help="绑定IP，默认0.0.0.0")
    parser.add_argument("--timeout", type=float, default=2.0, help="接收超时秒数")
    parser.add_argument("--skip_checksum", action="store_true", help="跳过累加和校验")

    # 发布器配置
    parser.add_argument("--out_group", default="", help="输出组播地址")
    parser.add_argument("--out_port", type=int, default=0, help="输出端口")
    parser.add_argument("--out_iface", default="0.0.0.0", help="输出网卡IP")
    parser.add_argument("--ttl", type=int, default=1, help="组播TTL")

    # 预测器配置
    parser.add_argument("--model_dir", default="", help="模型目录路径")
    parser.add_argument("--model_files", default="", help="模型文件名，逗号分隔")
    parser.add_argument("--meta_file", default="", help="Meta JSON 文件路径")
    parser.add_argument("--classes", default="", help="类别名，逗号分隔")

    # 缓冲区配置
    parser.add_argument("--seq_len", type=int, default=20, help="序列长度")
    parser.add_argument("--min_seq_len", type=int, default=20, help="最小序列长度")
    parser.add_argument("--max_age_s", type=float, default=10.0, help="轨迹最大保留秒数")
    parser.add_argument("--window_step", type=int, default=0, help="滑窗步长(0=每条都推理)")

    # 归一化配置
    parser.add_argument("--stats_path", default="", help="归一化统计文件路径")

    # 推理配置
    parser.add_argument("--publish_interval_ms", type=int, default=0, help="发布节流毫秒(0=不节流)")
    parser.add_argument("--print_targets", action="store_true", help="打印解析后的目标信息")
    parser.add_argument("--print_features", action="store_true", help="打印送入模型的特征值")
    parser.add_argument("--use_batch_ema", action="store_true", help="同批号目标使用EMA平滑概率并统一判别")
    parser.add_argument("--ema_alpha", type=float, default=0.6, help="EMA平滑系数(0-1)")

    # 本地测试配置
    parser.add_argument("--local_test", action="store_true", help="启用本地文件测试(替代组播)")
    parser.add_argument("--local_test_path", default="", help="本地 .xls/.csv 路径")
    parser.add_argument("--local_test_points", type=int, default=20, help="读取点数(默认前20)")

    return parser


def load_config(args: argparse.Namespace | None = None) -> argparse.Namespace:
    """加载配置，合并 YAML 文件和命令行参数。

    Args:
        args: 可选的已解析命令行参数，如果为 None 则自动解析。

    Returns:
        合并后的配置对象。
    """
    parser = create_argument_parser()
    parsed_args = args if args is not None else parser.parse_args()

    # 如果指定了配置文件，加载并合并配置
    if parsed_args.config:
        config = load_yaml_config(parsed_args.config)
        parsed_args = merge_config_with_args(config, parsed_args)

    return parsed_args
