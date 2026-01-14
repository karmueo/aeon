"""轨迹缓冲模块。

管理雷达航迹的时间窗口数据，支持滑窗推理和过期清理。
"""

import time
from collections import deque
from typing import Any

import numpy as np


class TrackWindowBuffer:
    """航迹窗口缓冲区。

    管理单个批号（batch_id）下所有航迹的时间序列数据。
    支持滑窗推理、自动清理过期数据等功能。
    """

    def __init__(self, seq_len: int, max_age_s: float = 10.0):
        """初始化航迹缓冲区。

        Args:
            seq_len: 序列窗口长度。
            max_age_s: 航迹最大保留时间（秒），超过此时间未更新的航迹将被清理。
        """
        self.seq_len = seq_len
        self.max_age_s = max_age_s

        # 航迹数据存储
        self.buffers: dict[int, deque] = {}  # track_id -> 特征向量队列
        self.last_seen: dict[int, float] = {}  # track_id -> 最后更新时间
        self.last_timestamp: dict[int, float] = {}  # track_id -> 最后时间戳
        self.since_last_infer: dict[int, int] = {}  # track_id -> 距上次推理的点数
        self.has_inferred: dict[int, bool] = {}  # track_id -> 是否已推理过

    def update(
        self, track_id: int, feature_vec: np.ndarray, timestamp_s: float | None = None
    ) -> None:
        """更新航迹数据。

        Args:
            track_id: 航迹 ID。
            feature_vec: 特征向量。
            timestamp_s: 时间戳（秒），可选。
        """
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.seq_len)
            self.since_last_infer[track_id] = 0
            self.has_inferred[track_id] = False

        self.buffers[track_id].append(feature_vec)
        self.last_seen[track_id] = time.time()
        self.since_last_infer[track_id] += 1

        if timestamp_s is not None:
            self.last_timestamp[track_id] = timestamp_s

    def cleanup(self) -> None:
        """清理过期的航迹数据。"""
        now = time.time()
        expired = [
            tid for tid, ts in self.last_seen.items() if now - ts > self.max_age_s
        ]
        for tid in expired:
            self.buffers.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.last_timestamp.pop(tid, None)
            self.since_last_infer.pop(tid, None)
            self.has_inferred.pop(tid, None)

    def build_batch(
        self, min_seq_len: int = 1, window_step: int = 0
    ) -> tuple[list[int], np.ndarray | None, np.ndarray | None]:
        """构建推理批次数据。

        Args:
            min_seq_len: 最小序列长度，低于此长度的航迹不参与推理。
            window_step: 滑窗步长，0 表示每条都推理，大于 0 表示每隔 N 点推理一次。

        Returns:
            (track_ids, batch, lengths)
            - track_ids: 航迹 ID 列表。
            - batch: 批次数据，形状为 (n_tracks, seq_len, n_features)，如果无数据则为 None。
            - lengths: 实际序列长度数组，形状为 (n_tracks,)，如果无数据则为 None。
        """
        track_ids = []
        sequences = []
        lengths = []

        for tid, buf in self.buffers.items():
            length = len(buf)
            if length < min_seq_len:
                continue

            # 滑窗控制
            if window_step > 0 and self.has_inferred.get(tid, False):
                if self.since_last_infer.get(tid, 0) < window_step:
                    continue

            # 构建序列
            seq = np.stack(list(buf), axis=0)

            # 截断或填充到 seq_len
            if length > self.seq_len:
                seq = seq[-self.seq_len :, :]
                length = self.seq_len
            if length < self.seq_len:
                pad = np.zeros((self.seq_len - length, seq.shape[1]), dtype=seq.dtype)
                seq = np.concatenate([seq, pad], axis=0)

            track_ids.append(tid)
            sequences.append(seq)
            lengths.append(length)

        if not sequences:
            return [], None, None

        batch = np.stack(sequences, axis=0)
        lengths = np.asarray(lengths, dtype=np.int16)
        return track_ids, batch, lengths

    def mark_inferred(self, track_ids: list[int]) -> None:
        """标记航迹已推理。

        Args:
            track_ids: 航迹 ID 列表。
        """
        for tid in track_ids:
            if tid in self.since_last_infer:
                self.since_last_infer[tid] = 0
                self.has_inferred[tid] = True

    def get_pending_progress(
        self, min_seq_len: int = 1, window_step: int = 0
    ) -> dict[int, tuple[int, int]]:
        """获取待推理航迹的进度信息。

        Args:
            min_seq_len: 最小序列长度。
            window_step: 滑窗步长。

        Returns:
            进度字典，键为航迹 ID，值为 (当前值, 目标值) 元组。
        """
        progress = {}
        for tid, buf in self.buffers.items():
            length = len(buf)
            if not self.has_inferred.get(tid, False):
                if length < min_seq_len:
                    progress[tid] = (length, min_seq_len)
            elif window_step > 0:
                delta = self.since_last_infer.get(tid, 0)
                if delta < window_step:
                    progress[tid] = (delta, window_step)
        return progress

    def get_last_timestamp(self, track_id: int) -> float | None:
        """获取航迹的最后时间戳。

        Args:
            track_id: 航迹 ID。

        Returns:
            最后时间戳（秒），如果不存在则返回 None。
        """
        return self.last_timestamp.get(track_id)

    @property
    def num_tracks(self) -> int:
        """获取当前缓冲的航迹数量。"""
        return len(self.buffers)
