"""会话持久化存储：配置、增量状态、结果、批次号全部落盘。

目录布局（``DATA_DIR/<session_id>/``）：

    config.json     阵列参数与估计配置（创建后只读）
    state.json      累加器状态 + 最新版本号（原子写）
    results.jsonl   每次估计结果一行（追加写，重启不丢）
    batches.log     已计入的批次号，一行一个（幂等去重依据）

所有 JSON 写盘走“临时文件 + fsync + os.replace”原子替换/追加，
避免崩溃时写出半截文件。JSONL/日志文件以追加模式打开并 fsync。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class SessionStore:
    def __init__(self, data_dir: str | os.PathLike):
        self.root = Path(data_dir)
        # 目录惰性创建：仅实例化不落盘（测试/未挂载场景不触碰 /data），
        # 首次 create 时再建。
        self._lock = threading.Lock()

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def list_sessions(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and (p / "config.json").exists()
        )

    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def exists(self, session_id: str) -> bool:
        return (self.root / session_id / "config.json").exists()

    # ------------------------------------------------------------------
    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _append_jsonl(path: Path, rows: Iterable[dict]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _append_lines(path: Path, lines: Iterable[str]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(str(line) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------------
    def create(self, session_id: str, config: dict) -> None:
        sdir = self.root / session_id
        with self._lock:
            self._ensure_root()
            if sdir.exists():
                raise FileExistsError(session_id)
            sdir.mkdir(parents=True)
            try:
                self._atomic_write_json(sdir / "config.json", config)
                self._atomic_write_json(
                    sdir / "state.json",
                    {"version": 0, "latest_result": None},
                )
                (sdir / "results.jsonl").touch()
                (sdir / "batches.log").touch()
            except BaseException:
                # 创建失败不残留半成品目录
                for p in sorted(sdir.glob("**/*"), reverse=True):
                    if p.is_file():
                        p.unlink()
                sdir.rmdir()
                raise

    def load_config(self, session_id: str) -> dict:
        with open(self.root / session_id / "config.json", encoding="utf-8") as f:
            return json.load(f)

    def load_state(self, session_id: str) -> dict:
        path = self.root / session_id / "state.json"
        if not path.exists():
            return {"version": 0, "latest_result": None}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save_state(self, session_id: str, state: dict) -> None:
        self._atomic_write_json(self.root / session_id / "state.json", state)

    def append_results(self, session_id: str, rows: list[dict]) -> None:
        self._append_jsonl(self.root / session_id / "results.jsonl", rows)

    def load_results(self, session_id: str) -> list[dict]:
        path = self.root / session_id / "results.jsonl"
        if not path.exists():
            return []
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def load_batch_ids(self, session_id: str) -> set[str]:
        path = self.root / session_id / "batches.log"
        if not path.exists():
            return set()
        with open(path, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def record_batch_ids(self, session_id: str, batch_ids: list[str]) -> None:
        self._append_lines(self.root / session_id / "batches.log", batch_ids)

    # ------------------------------------------------------------------
    def save_buffer_npz(
        self, session_id: str, arrays: dict
    ) -> None:
        """原子写入滑动窗口环形缓冲（npz 二进制，避免大 JSON）。"""
        target = self.root / session_id / "buffer.npz"
        tmp = target.with_suffix(".npz.tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)

    def load_buffer_npz(self, session_id: str) -> dict | None:
        path = self.root / session_id / "buffer.npz"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            with np.load(f) as data:
                return {k: data[k].copy() for k in data.files}
