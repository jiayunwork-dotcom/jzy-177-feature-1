"""HTTP 请求/响应的 Pydantic 模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ArrayConfigModel(BaseModel):
    elements: int = Field(..., ge=2, le=64, description="阵元数 M（2..64）")
    spacing: float = Field(..., gt=0, description="阵元间距（米），必须为正")
    wavelength: float = Field(..., gt=0, description="载波波长（米），必须为正")
    position_errors: list[float] | None = Field(
        None, description="可选阵元轴向位置误差（米），长度 M"
    )


class EstimationConfigModel(BaseModel):
    angle_min: float = -90.0
    angle_max: float = 90.0
    angle_step: float = Field(1.0, gt=0)
    smoothing: bool = False
    subarray_length: int | None = Field(None, ge=2)
    source_count: int | None = Field(None, ge=1, description="为空时用 MDL（另报 AIC）")


class UpdateConfigModel(BaseModel):
    mode: Literal["sliding", "forgetting"] = "sliding"
    window_size: int | None = Field(500, ge=1)
    forgetting_factor: float | None = Field(None, gt=0.0, le=1.0)


class CreateSessionModel(BaseModel):
    array: ArrayConfigModel
    estimation: EstimationConfigModel | None = None
    update: UpdateConfigModel | None = None


class AppendModel(BaseModel):
    batch_id: str = Field(..., min_length=1, description="批次号，重复提交只计一次")
    real: list[list[float]]
    imag: list[list[float]]
    estimate: bool = Field(
        False, description="追加成功后是否立即按当前配置出一次估计"
    )
    # 允许本次估计临时覆盖会话配置（不写回会话配置）
    source_count: int | None = Field(None, ge=1)
    smoothing: bool | None = None
    subarray_length: int | None = Field(None, ge=2)
    angle_min: float | None = None
    angle_max: float | None = None
    angle_step: float | None = Field(None, gt=0)
    true_angles: list[float] | None = Field(
        None, description="可选真实方位（度），用于回传每角偏差与 偏差/CRB 比值"
    )


class EstimateNowModel(BaseModel):
    source_count: int | None = Field(None, ge=1)
    smoothing: bool | None = None
    subarray_length: int | None = Field(None, ge=2)
    angle_min: float | None = None
    angle_max: float | None = None
    angle_step: float | None = Field(None, gt=0)
    true_angles: list[float] | None = None


class SourceModel(BaseModel):
    angle: float
    power: float = Field(..., gt=0)
    coherent: bool = False


class SynthesisModel(BaseModel):
    elements: int = Field(..., ge=2, le=64)
    spacing: float = Field(..., gt=0)
    wavelength: float = Field(..., gt=0)
    position_errors: list[float] | None = None
    k_snapshots: int = Field(..., ge=1, le=10_000)
    snr_db: float
    seed: int
    sources: list[SourceModel]
    # 直接对合成数据跑一次估计
    estimate: bool = True
    angle_min: float = -90.0
    angle_max: float = 90.0
    angle_step: float = Field(1.0, gt=0)
    smoothing: bool = False
    subarray_length: int | None = Field(None, ge=2)
    source_count: int | None = Field(None, ge=1)
    include_data: bool = Field(
        False, description="是否回传合成的实部/虚部（默认只给估计结果）"
    )
