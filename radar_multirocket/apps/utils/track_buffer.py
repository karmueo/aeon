"""航迹缓冲管理模块。

管理多批号多航迹的滑动窗口数据，支持变长序列和超时清理。
"""

import time
from collections import defaultdict
from typing import Any

import numpy as np


class TrackWindowBuffer:
    """航迹滑动窗口缓冲区。

    为每个批号-航迹对维护固定长度的特征序列缓冲。
    """

    def __init__(self, seq_len: int, max_age_s: float = 10.0):
        """初始化缓冲区。

        Args:
            seq_len: 序列长度。
            max_age_s: 轨迹最大保留时间（秒）。
        """
        self.seq_len = seq_len
        self.max_age_s = max_age_s
        # {track_id: {"features": deque, "last_seen": float, "inferred": bool}}
        self._tracks: dict[int, dict[str, Any]] = {}
        self._next_track_id = 1

    def update(
        self, track_id: int | None, features: np.ndarray, timestamp_s: float | None = None
    ) -> int:
        """更新航迹特征。

        Args:
            track_id: 航迹 ID，为 None 时自动分配。
            features: 特征向量，形状为 (n_features,)。
            timestamp_s: 时间戳（秒）。

        Returns:
            分配或使用的航迹 ID。
        """
        now = timestamp_s if timestamp_s is not None else time.time()

        if track_id is None or track_id <= 0:
            track_id = self._next_track_id
            self._next_track_id += 1

        track = self._tracks.get(track_id)
        if track is None:
            self._tracks[track_id] = {
                "features": [],
                "last_seen": now,
                "inferred": False,
            }
            track = self._tracks[track_id]

        track["features"].append(features)
        # 保留最近的 seq_len 个特征
        if len(track["features"]) > self.seq_len:
            track["features"] = track["features"][-self.seq_len :]
        track["last_seen"] = now
        track["inferred"] = False

        return track_id

    def cleanup(self) -> None:
        """清理过期航迹。"""
        now = time.time()
        expired = [
            tid
            for tid, track in self._tracks.items()
            if now - track["last_seen"] > self.max_age_s
        ]
        for tid in expired:
            del self._tracks[tid]

    def get_pending_progress(
        self, min_seq_len: int, window_step: int = 0
    ) -> dict[int, tuple[int, int]]:
        """获取待推理航迹的进度。

        Args:
            min_seq_len: 最小序列长度。
            window_step: 滑窗步长。

        Returns:
            {track_id: (current_len, needed_len)} 进度字典。
        """
        pending = {}
        for tid, track in self._tracks.items():
            if track["inferred"]:
                continue
            current_len = len(track["features"])
            if window_step == 0:
                # 滑窗步长为0时，显示所有未达到最小序列长度的轨迹进度
                if current_len < min_seq_len:
                    pending[tid] = (current_len, min_seq_len)
            else:
                # 滑窗步长>0时，显示达到最小长度但未达到完整长度的轨迹
                if current_len >= min_seq_len and current_len < self.seq_len:
                    pending[tid] = (current_len, self.seq_len)
        return pending

    def build_batch(
        self, min_seq_len: int, window_step: int = 0
    ) -> tuple[list[int], np.ndarray | None, np.ndarray | None]:
        """构建批量推理数据。

        Args:
            min_seq_len: 最小序列长度。
            window_step: 滑窗步长，0 表示只在达到 seq_len 时推理一次。

        Returns:
            (track_ids, batch, lengths)
            - track_ids: 航迹 ID 列表。
            - batch: 形状为 (n_samples, seq_len, n_features) 的批量数据。
            - lengths: 形状为 (n_samples,) 的实际序列长度。
        """
        track_ids = []
        sequences = []
        lengths = []

        for tid, track in self._tracks.items():
            if track["inferred"]:
                continue

            feats = track["features"]
            current_len = len(feats)

            # 检查是否满足推理条件
            if window_step == 0:
                # 只在达到 seq_len 时推理一次
                if current_len < self.seq_len:
                    continue
                seq = feats[-self.seq_len :]
                sequences.append(seq)
                lengths.append(self.seq_len)
                track_ids.append(tid)
            else:
                # 滑窗模式：每 window_step 个点推理一次
                if current_len < min_seq_len:
                    continue
                # 从 min_seq_len 开始，每隔 window_step 取一个窗口
                start_idx = min_seq_len
                while start_idx <= current_len:
                    end_idx = min(start_idx, current_len)
                    # 取最近的 seq_len 个点，不足则从开头补齐
                    if end_idx >= self.seq_len:
                        seq = feats[end_idx - self.seq_len : end_idx]
                    else:
                        seq = feats[:end_idx]
                        # 前补零到 seq_len
                        pad_len = self.seq_len - len(seq)
                        pad = np.zeros((pad_len, len(seq[0])), dtype=np.float32)
                        seq = np.vstack([pad, seq])
                    sequences.append(seq)
                    lengths.append(len(seq))
                    track_ids.append(tid)
                    start_idx += window_step

        if not sequences:
            return track_ids, None, None

        batch = np.stack(sequences, axis=0)
        lengths = np.array(lengths, dtype=np.int64)
        return track_ids, batch, lengths

    def get_last_timestamp(self, track_id: int) -> float | None:
        """获取航迹最后时间戳。"""
        track = self._tracks.get(track_id)
        return track["last_seen"] if track else None

    def mark_inferred(self, track_ids: list[int] | int) -> None:
        """标记航迹已推理。"""
        if isinstance(track_ids, int):
            track_ids = [track_ids]
        for tid in track_ids:
            track = self._tracks.get(tid)
            if track:
                track["inferred"] = True
