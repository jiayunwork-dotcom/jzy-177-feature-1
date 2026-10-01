"""业务错误类型。

所有内核/会话层可预见的错误都继承 :class:`DOAError`，HTTP 层统一翻译为
422 响应，``type`` 字段给出机读错误码，``message`` 给出中文人读说明。
"""

from __future__ import annotations


class DOAError(Exception):
    """内核与人读错误的基类。"""

    type: str = "bad_request"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(DOAError):
    """参数或数据形状/取值非法。"""

    type = "validation_error"


class LimitExceededError(DOAError):
    """超出规模上限（阵元数 / 单批快拍数 / 扫描点数）。"""

    type = "limit_exceeded"


class InsufficientDataError(DOAError):
    """累计快拍不足以支撑估计（少于阵元数等）。"""

    type = "insufficient_data"


class SourceCountError(DOAError):
    """信源数与阵列/平滑配置不兼容。"""

    type = "source_count_error"


class SessionNotFoundError(DOAError):
    """会话不存在。"""

    type = "session_not_found"


class VersionConflictError(DOAError):
    """批次号回退等会话状态冲突。"""

    type = "version_conflict"
