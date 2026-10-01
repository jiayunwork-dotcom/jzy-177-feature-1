"""全局配置：会话数据目录等，可用环境变量覆盖。"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    data_dir: str = os.environ.get("DOA_DATA_DIR", "/data")
    default_window_size: int = int(os.environ.get("DOA_WINDOW_SIZE", "500"))
    default_forgetting_factor: float = float(
        os.environ.get("DOA_FORGETTING_FACTOR", "0.98")
    )


settings = Settings()
