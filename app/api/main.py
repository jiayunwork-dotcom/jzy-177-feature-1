"""FastAPI 应用与路由：会话管理、流式追加、轮询、SSE 订阅、合成自测。"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from ..config import settings
from ..doa.engine import (
    ArraySpec,
    EstConfig,
    MAX_SCAN_POINTS,
    MAX_SNAPSHOTS_PER_BATCH,
    estimate,
    validate_array_inputs,
)
from ..doa.synthesis import generate_snapshots
from ..errors import DOAError
from ..sessions.events import EventBroker
from ..sessions.session import SessionManager
from .schemas import (
    AppendModel,
    CreateSessionModel,
    EstimateNowModel,
    SynthesisModel,
)

broker = EventBroker()
manager = SessionManager(settings.data_dir, broker)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 重启后不预载全部会话：首次访问时懒加载（历史结果已在盘上）。
    yield


app = FastAPI(
    title="ULA-MUSIC 常驻测向后端",
    version="1.0.0",
    description=(
        "八阵元（上限 64）均匀线阵 MUSIC 测向常驻服务：流式增量协方差、"
        "MDL/AIC、前后向空间平滑、谱峰细化、单源 CRB、合成自测、"
        "会话持久化、结果轮询与 SSE 推送。"
    ),
    lifespan=lifespan,
)


@app.exception_handler(DOAError)
async def doa_error_handler(request: Request, exc: DOAError):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": exc.type,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sessions")
def create_session(body: CreateSessionModel):
    config = body.model_dump()
    session = manager.create_session(config)
    return session.status()


@app.get("/sessions")
def list_sessions():
    return {"sessions": manager.list_sessions()}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    return manager.get(session_id).status()


@app.post("/sessions/{session_id}/append")
def append_batch(session_id: str, body: AppendModel):
    real = np.asarray(body.real, dtype=np.float64)
    imag = np.asarray(body.imag, dtype=np.float64)
    if real.size > 0 and real.shape[1] > MAX_SNAPSHOTS_PER_BATCH:
        from ..errors import LimitExceededError

        raise LimitExceededError(
            f"单批快拍数 {real.shape[1]} 超过上限 {MAX_SNAPSHOTS_PER_BATCH}",
            details={"value": int(real.shape[1]), "max": MAX_SNAPSHOTS_PER_BATCH},
        )
    # 形状/NaN 校验在内核统一给出中文错误
    validate_array_inputs(real, imag)
    overrides = {
        "source_count": body.source_count,
        "smoothing": body.smoothing,
        "subarray_length": body.subarray_length,
        "angle_min": body.angle_min,
        "angle_max": body.angle_max,
        "angle_step": body.angle_step,
        "true_angles": body.true_angles,
    }
    session = manager.get(session_id)
    return session.append(
        body.batch_id,
        real,
        imag,
        run_estimate=body.estimate,
        overrides=overrides,
    )


@app.post("/sessions/{session_id}/estimate")
def estimate_now(session_id: str, body: EstimateNowModel | None = None):
    overrides = (
        body.model_dump()
        if body is not None
        else None
    )
    return manager.get(session_id).estimate(overrides)


@app.get("/sessions/{session_id}/results")
def poll_results(session_id: str, since_version: int = 0):
    """带版本号轮询：只返回严格比 ``since_version`` 新的结果。"""
    session = manager.get(session_id)
    rows = session.results_since(since_version)
    return {
        "session_id": session_id,
        "since_version": since_version,
        "latest_version": session.status()["version"],
        "results": rows,
    }


@app.get("/sessions/{session_id}/results/latest")
def latest_result(session_id: str):
    result = manager.get(session_id).latest_result()
    return {"session_id": session_id, "result": result}


@app.get("/sessions/{session_id}/stream")
async def stream_results(session_id: str, after_version: int = 0,
                         max_idle_seconds: float = 0.0):
    """SSE 订阅：先补发比 ``after_version`` 新的历史结果，再实时推送。

    默认（``max_idle_seconds=0``）持续推送直到客户端断开；
    自动化测试可给一个正的空闲超时，历史补发完且无新结果时流正常结束，
    避免无限心跳阻塞 HTTP 客户端的连接收尾。
    """
    session = manager.get(session_id)
    queue = broker.subscribe(session_id)
    history = session.results_since(after_version)

    async def event_gen():
        try:
            for row in history:
                yield f"event: result\ndata: {json.dumps(row, ensure_ascii=False)}\n\n"
            while True:
                wait_timeout = max_idle_seconds if max_idle_seconds > 0 else 15.0
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=wait_timeout)
                    if payload["result"]["version"] > after_version:
                        yield (
                            "event: result\ndata: "
                            + json.dumps(payload["result"], ensure_ascii=False)
                            + "\n\n"
                        )
                except asyncio.TimeoutError:
                    if max_idle_seconds > 0:
                        return
                    yield ": heartbeat\n\n"
        finally:
            broker.unsubscribe(session_id, queue)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ----------------------------------------------------------------------
@app.post("/synthesis")
def synthesis(body: SynthesisModel):
    """合成数据闭环自测：造数据（同种子逐位一致）并可选立即估计。"""
    spec = ArraySpec(
        elements=body.elements,
        spacing=body.spacing,
        wavelength=body.wavelength,
        position_errors=tuple(body.position_errors) if body.position_errors else None,
    )
    positions = spec.positions()
    syn = generate_snapshots(
        m=body.elements,
        k=body.k_snapshots,
        angles_deg=[s.angle for s in body.sources],
        powers=[s.power for s in body.sources],
        snr_db=body.snr_db,
        positions=positions,
        wavelength=body.wavelength,
        seed=body.seed,
        coherent=all(s.coherent for s in body.sources) and len(body.sources) > 0,
    )
    response = {
        "synthesis": {
            "seed": body.seed,
            "true_angles": syn["angles"],
            "powers": syn["powers"],
            "noise_variance": syn["noise_variance"],
            "k_snapshots": body.k_snapshots,
            "coherent": syn["coherent"],
        }
    }
    if body.include_data:
        response["data"] = {
            "real": np.real(syn["data"]).tolist(),
            "imag": np.imag(syn["data"]).tolist(),
        }
    if body.estimate:
        cfg = EstConfig(
            angle_min=body.angle_min,
            angle_max=body.angle_max,
            angle_step=body.angle_step,
            smoothing=body.smoothing,
            subarray_length=body.subarray_length,
            source_count=body.source_count,
        )
        response["result"] = estimate(
            syn["data"],
            spec,
            cfg,
            true_angles=syn["angles"],
        )
    return response
