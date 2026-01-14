"""异步报文记录器模块。

使用异步队列和后台线程实现高性能报文记录，确保不阻塞推理主循环。
"""

import csv
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


class PacketRecorder:
    """异步报文记录器。

    使用后台线程和队列缓冲，实现高性能的报文记录，不影响推理性能。
    """

    def __init__(self, output_dir: str, format_type: str = "csv"):
        """初始化报文记录器。

        Args:
            output_dir: 输出目录路径。
            format_type: 记录格式，"csv" 或 "xls"。
        """
        self.output_dir = Path(output_dir)
        self.format_type = format_type.lower()
        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._batch_buffers: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._batch_recording_started: set[int] = set()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        # 验证格式类型
        if self.format_type not in ("csv", "xls"):
            raise ValueError(f"不支持的记录格式: {format_type}，仅支持 csv 或 xls")

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """启动后台记录线程。"""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="PacketRecorder"
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """停止后台记录线程，并等待所有数据写入完成。"""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            # 等待队列处理完成（最多等待5秒）
            self._worker_thread.join(timeout=5.0)
        self._flush_all_batches()

    def add_packet(
        self, batch_id: int, target: dict[str, Any], min_seq_len: int, current_seq_len: int
    ) -> None:
        """添加报文到记录队列。

        Args:
            batch_id: 批号。
            target: 目标报文数据。
            min_seq_len: 最小序列长度要求。
            current_seq_len: 当前序列长度。
        """
        try:
            # 非阻塞式添加，避免满队列时阻塞主循环
            self._queue.put_nowait((batch_id, target, min_seq_len, current_seq_len))
        except queue.Full:
            # 队列满时丢弃最旧的数据
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((batch_id, target, min_seq_len, current_seq_len))
            except queue.Empty:
                pass

    def finish_batch(self, batch_id: int) -> None:
        """标记批号结束，触发记录保存。

        Args:
            batch_id: 批号。
        """
        try:
            self._queue.put_nowait(("finish", batch_id, None, None))
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        """后台工作线程主循环。"""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
                cmd, batch_id, target, seq_info = item

                if cmd == "finish":
                    self._save_batch(batch_id)
                else:
                    min_seq_len, current_seq_len = seq_info, target.get("seq_len", 0)

                    # 检查是否满足最小序列长度
                    if current_seq_len >= min_seq_len:
                        if batch_id not in self._batch_recording_started:
                            self._batch_recording_started.add(batch_id)
                        self._batch_buffers[batch_id].append(target)

            except queue.Empty:
                continue
            except Exception as e:
                # 记录错误但不中断线程
                print(f"[PacketRecorder] 错误: {e}")

    def _save_batch(self, batch_id: int) -> None:
        """保存批号数据到文件。

        Args:
            batch_id: 批号。
        """
        if batch_id not in self._batch_buffers:
            return

        data = self._batch_buffers.pop(batch_id, [])
        self._batch_recording_started.discard(batch_id)

        if not data:
            return

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_{batch_id}_{timestamp}.{self.format_type}"
        filepath = self.output_dir / filename

        try:
            if self.format_type == "csv":
                self._write_csv(filepath, data)
            else:  # xls
                self._write_xls(filepath, data)
        except Exception as e:
            print(f"[PacketRecorder] 保存文件失败 {filepath}: {e}")

    def _write_csv(self, filepath: Path, data: list[dict[str, Any]]) -> None:
        """写入 CSV 文件。

        Args:
            filepath: 文件路径。
            data: 数据列表。
        """
        if not data:
            return

        # 收集所有字段
        fieldnames = set()
        for item in data:
            fieldnames.update(item.keys())
        fieldnames = sorted(fieldnames)

        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def _write_xls(self, filepath: Path, data: list[dict[str, Any]]) -> None:
        """写入 XLS (Tab分隔) 文件。

        Args:
            filepath: 文件路径。
            data: 数据列表。
        """
        if not data:
            return

        # 收集所有字段
        fieldnames = set()
        for item in data:
            fieldnames.update(item.keys())
        fieldnames = sorted(fieldnames)

        with filepath.open("w", newline="", encoding="utf-8") as f:
            # 写入表头
            f.write("\t".join(fieldnames) + "\n")
            # 写入数据
            for item in data:
                row = [str(item.get(field, "")) for field in fieldnames]
                f.write("\t".join(row) + "\n")

    def _flush_all_batches(self) -> None:
        """刷新所有未保存的批号数据。"""
        for batch_id in list(self._batch_buffers.keys()):
            self._save_batch(batch_id)

    @property
    def is_recording(self) -> bool:
        """检查是否正在记录。"""
        return self._worker_thread is not None and self._worker_thread.is_alive()

    @property
    def pending_count(self) -> int:
        """获取队列中待处理的报文数量。"""
        return self._queue.qsize()
