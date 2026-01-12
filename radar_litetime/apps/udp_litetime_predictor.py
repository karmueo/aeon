"""UDP 组播航迹 LITETime 分类应用。

实现接收 UDP 组播报文、解析航迹数据、使用 LITETime 模型推理、发布结果。
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# 添加 Time-Series-Library 到路径以复用 UDP 模块
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TSL_ROOT = PROJECT_ROOT / "Time-Series-Library"

if str(TSL_ROOT) not in sys.path:
    sys.path.insert(0, str(TSL_ROOT))

from udp.parser import parse_packet
from udp.publisher import MulticastPublisher
from udp.receiver import MulticastReceiver

# 添加 radar_litetime 到路径以导入 apps 模块
RADAR_ROOT = Path(__file__).resolve().parents[1]
if str(RADAR_ROOT) not in sys.path:
    sys.path.insert(0, str(RADAR_ROOT))

from apps.config.loader import load_config
from apps.core.buffer import TrackWindowBuffer
from apps.core.predictor import PredictorFactory
from apps.utils.feature_processor import FeatureExtractor, FeatureNormalizer


def _detect_delimiter(path: Path) -> str:
    """检测文件分隔符。"""
    return "\t" if path.suffix.lower() == ".xls" else ","


def _parse_float(value: str, field: str, line_no: int) -> float:
    """解析浮点数值。"""
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"第{line_no}行字段{field}无法转换为浮点数: {value}") from exc


def load_local_trajectory(
    path: str, max_points: int, feature_extractor: FeatureExtractor
) -> list[np.ndarray]:
    """加载本地轨迹文件用于测试。

    Args:
        path: 文件路径。
        max_points: 最大读取点数。
        feature_extractor: 特征提取器。

    Returns:
        特征向量列表。
    """
    if max_points <= 0:
        raise ValueError("local_test_points 必须大于 0")

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"本地文件不存在: {file_path}")

    delimiter = _detect_delimiter(file_path)
    features = []
    with file_path.open("r", encoding="gbk") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("本地文件缺少表头")

        for line_no, row in enumerate(reader, start=2):
            if len(features) >= max_points:
                break

            # 构造目标字典
            target = {}
            for key, value in row.items():
                value = value.strip() if isinstance(value, str) else value
                if value in ("", None):
                    target[key] = 0.0
                else:
                    target[key] = _parse_float(str(value), key, line_no)

            # 特殊处理单位转换
            if "r_m" in target or "径向距离" in target:
                key = "r_m" if "r_m" in target else "径向距离"
                target[key] = float(target[key]) / 1000.0
            if "pr_m" in target or "点迹距离" in target:
                key = "pr_m" if "pr_m" in target else "点迹距离"
                target[key] = float(target[key]) / 1000.0

            # 提取特征
            feat = feature_extractor.extract(target)
            if feat is not None:
                features.append(feat)

    if not features:
        raise ValueError("本地文件无有效数据行")
    return features


def run_local_file_inference(
    args, feature_extractor: FeatureExtractor, predictor
) -> None:
    """运行本地文件推理测试。

    Args:
        args: 命令行参数。
        feature_extractor: 特征提取器。
        predictor: 预测器。
    """
    local_features = load_local_trajectory(
        args.local_test_path, args.local_test_points, feature_extractor
    )
    if len(local_features) < args.seq_len:
        raise ValueError(
            f"本地文件点数不足: {len(local_features)} < seq_len={args.seq_len}"
        )

    seq = np.stack(local_features[: args.seq_len], axis=0).astype(np.float32)
    seq = np.asarray([seq], dtype=np.float32)
    lengths = np.asarray([args.seq_len], dtype=np.int64)

    preds, probs = predictor.predict(seq, lengths)
    prob_uav = (
        float(probs[0, 1])
        if probs.shape[1] > 1
        else float(probs[0, 0])
    )
    prob_bird = (
        float(probs[0, 0])
        if probs.shape[1] > 1
        else 1.0 - prob_uav
    )

    item = {
        "batch_id": 1,
        "track_id": 1,
        "timestamp_ms": 0,
        "time_25us": 0,
        "pred": int(preds[0]),
        "prob_uav": prob_uav,
        "prob_bird": prob_bird,
    }
    print(
        json.dumps(
            {"batch_id": 1, "count": 1, "items": [item]}, ensure_ascii=False
        )
    )


def run_multicast_inference(
    args, feature_extractor: FeatureExtractor, predictor
) -> None:
    """运行 UDP 组播推理主循环。

    Args:
        args: 命令行参数。
        feature_extractor: 特征提取器。
        predictor: 预测器。
    """
    # 初始化 UDP 组件
    receiver = (
        MulticastReceiver(
            group=args.in_group,
            port=args.in_port,
            iface=args.in_iface,
            bind_ip=args.bind_ip,
            timeout_s=args.timeout,
        )
        .open()
    )
    publisher = (
        MulticastPublisher(
            group=args.out_group,
            port=args.out_port,
            iface=args.out_iface,
            ttl=args.ttl,
        )
        .open()
    )

    # 批号缓冲管理
    batch_buffers = {}
    batch_last_seen = {}
    batch_last_publish_ts = {}
    batch_ema_prob_uav = {}

    min_seq_len = max(1, min(args.min_seq_len, args.seq_len))

    print("开始接收组播并预测...")

    while True:
        try:
            data, addr = receiver.recv()
        except Exception:
            continue

        # 解析报文
        targets = parse_packet(data, skip_checksum=args.skip_checksum)
        if not targets:
            continue

        if args.print_targets:
            out = {
                "src": f"{addr[0]}:{addr[1]}",
                "count": len(targets),
                "targets": targets,
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))

        # 处理每个目标
        for tar in targets:
            # 提取特征
            feat = feature_extractor.extract(tar)
            if feat is None:
                continue

            if args.print_features:
                print(f"Feature shape: {feat.shape}, values: {feat}")

            # 获取批号和航迹 ID
            batch_id = tar.get(
                "目标批号", tar.get("batch_id", tar.get("track_id", 0))
            )
            track_id = tar.get("track_id", tar.get("tar_seq", batch_id))

            # 更新缓冲
            buffer = batch_buffers.get(batch_id)
            if buffer is None:
                buffer = TrackWindowBuffer(
                    seq_len=args.seq_len, max_age_s=args.max_age_s
                )
                batch_buffers[batch_id] = buffer
            batch_last_seen[batch_id] = time.time()
            buffer.update(track_id, feat, timestamp_s=tar.get("timestamp"))

        # 清理过期批号
        now = time.time()
        expired_batches = [
            bid for bid, ts in batch_last_seen.items() if now - ts > args.max_age_s
        ]
        for bid in expired_batches:
            batch_buffers.pop(bid, None)
            batch_last_seen.pop(bid, None)
            batch_last_publish_ts.pop(bid, None)
            batch_ema_prob_uav.pop(bid, None)

        # 批量推理
        all_batch_ids = []
        all_track_ids = []
        all_buffers = []

        for batch_id, buffer in list(batch_buffers.items()):
            buffer.cleanup()
            pending = buffer.get_pending_progress(
                min_seq_len=min_seq_len, window_step=args.window_step
            )
            if pending:
                for tid, (count, needed) in pending.items():
                    print(f"batch_id={batch_id} track_id={tid} [{count}/{needed}]")

            track_ids, batch, lengths = buffer.build_batch(
                min_seq_len=min_seq_len, window_step=args.window_step
            )
            if batch is None:
                continue

            # 发布节流
            if args.publish_interval_ms > 0:
                last_ts = batch_last_publish_ts.get(batch_id, 0.0)
                if (now - last_ts) * 1000 < args.publish_interval_ms:
                    continue

            all_batch_ids.append(batch_id)
            all_track_ids.append(track_ids)
            all_buffers.append((batch, lengths, buffer))

        # 批量推理
        if all_buffers:
            merged_batch = np.concatenate([b[0] for b in all_buffers], axis=0)
            merged_lengths = np.concatenate([b[1] for b in all_buffers], axis=0)

            inference_start = time.time()
            preds, probs = predictor.predict(merged_batch, merged_lengths)
            inference_time_ms = (time.time() - inference_start) * 1000

            # 分发结果
            sample_offset = 0
            for idx, batch_id in enumerate(all_batch_ids):
                track_ids = all_track_ids[idx]
                buffer = all_buffers[idx][2]
                num_samples = len(track_ids)

                batch_preds = preds[sample_offset : sample_offset + num_samples]
                batch_probs = probs[sample_offset : sample_offset + num_samples]

                items = []
                for i, tid in enumerate(track_ids):
                    prob_uav = (
                        float(batch_probs[i, 1])
                        if batch_probs.shape[1] > 1
                        else float(batch_probs[i, 0])
                    )
                    prob_bird = (
                        float(batch_probs[i, 0])
                        if batch_probs.shape[1] > 1
                        else 1.0 - prob_uav
                    )
                    last_ts = float(buffer.get_last_timestamp(tid) or 0.0)
                    items.append({
                        "batch_id": batch_id,
                        "track_id": tid,
                        "timestamp_ms": int(last_ts * 1000),
                        "time_25us": int(round(last_ts / 25e-6)),
                        "pred": int(batch_preds[i]),
                        "prob_uav": prob_uav,
                        "prob_bird": prob_bird,
                    })

                # EMA 平滑
                if args.use_batch_ema and items:
                    alpha = max(0.0, min(1.0, float(args.ema_alpha)))
                    current_prob = float(
                        np.mean([item["prob_uav"] for item in items])
                    )
                    prev_prob = batch_ema_prob_uav.get(batch_id)
                    ema_prob = (
                        current_prob
                        if prev_prob is None
                        else alpha * current_prob + (1.0 - alpha) * prev_prob
                    )
                    batch_ema_prob_uav[batch_id] = ema_prob
                    batch_pred = 1 if ema_prob >= 0.5 else 0
                    for item in items:
                        item["pred"] = batch_pred
                        item["prob_uav"] = ema_prob
                        item["prob_bird"] = 1.0 - ema_prob

                print(
                    json.dumps(
                        {
                            "batch_id": batch_id,
                            "count": len(items),
                            "inference_time_ms": round(inference_time_ms, 2),
                            "items": items,
                        },
                        ensure_ascii=False,
                    )
                )
                publisher.send(items)
                buffer.mark_inferred(track_ids)
                batch_last_publish_ts[batch_id] = time.time()

                sample_offset += num_samples


def main() -> None:
    """主入口函数。"""
    # 加载配置
    args = load_config()

    # 参数验证
    if args.local_test:
        if not args.local_test_path:
            raise ValueError("启用 --local_test 时必须提供 --local_test_path")
        if args.local_test_points <= 0:
            raise ValueError("--local_test_points 必须大于 0")
    else:
        if not args.in_group:
            raise ValueError("未启用本地测试时必须提供 --in_group")
        if args.in_port <= 0:
            raise ValueError("未启用本地测试时必须提供 --in_port")
        if not args.out_group:
            raise ValueError("未启用本地测试时必须提供 --out_group")
        if args.out_port <= 0:
            raise ValueError("未启用本地测试时必须提供 --out_port")

    # 初始化特征提取器
    normalizer = (
        FeatureNormalizer.from_stats_file(args.stats_path)
        if args.stats_path
        else FeatureNormalizer()
    )
    feature_extractor = FeatureExtractor(normalizer=normalizer)

    # 初始化预测器
    if args.meta_file:
        predictor = PredictorFactory.create_from_meta(
            meta_path=args.meta_file, model_dir=args.model_dir
        )
    else:
        if not args.model_files or not args.classes:
            raise ValueError("必须提供 --meta_file 或 (--model_files 和 --classes)")
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        predictor = PredictorFactory.create(
            model_files=args.model_files, classes=classes, model_dir=args.model_dir
        )

    predictor.load(num_features=feature_extractor.num_features)

    # 运行推理
    if args.local_test:
        run_local_file_inference(args, feature_extractor, predictor)
    else:
        run_multicast_inference(args, feature_extractor, predictor)


if __name__ == "__main__":
    main()
