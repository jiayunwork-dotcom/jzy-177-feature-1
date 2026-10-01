# ULA-MUSIC 常驻测向后端

把八阵元（可扩到 64 阵元）均匀线阵的 MUSIC 测向变成常驻 HTTP 服务：
上位机按批流式推送复基带快拍，服务增量维护协方差、估计信源数
（MDL/AIC）、做空间平滑与 MUSIC 谱扫描细化、逐角给出单源白噪声
确定性模型 CRB，并支持会话持久化、版本轮询与 SSE 订阅推送。

测向内核（协方差、增量更新、源数判定、空间平滑、谱扫描与细化、
CRB、数据合成）全部用 NumPy 自行实现，未使用任何现成测向工具包。

---

## 1. 目录结构（按职责拆模块）

```
app/
  errors.py                 # 人读业务错误（统一翻成 422 + 中文 message + 机读 type）
  config.py                 # 环境变量配置（数据目录、默认窗口/遗忘因子）
  doa/                      # —— 估计内核（不依赖会话与 HTTP）——
    array.py                # 阵元位置、方向向量、栅瓣模糊角度对
    covariance.py           # 批量样本协方差 / 厄米化
    accumulator(在 sessions)# ——
    smoothing.py            # 前后向空间平滑、子阵长度校验（可分辨上限）
    criteria.py             # MDL / AIC（Wax-Kailath 形式）
    music.py                # 噪声子空间、空间谱、找峰、黄金分割细化
    crb.py                  # 单源白噪声确定性模型 CRB
    synthesis.py            # 合成数据（PCG64，同种子逐位一致）
    engine.py               # 总装 + 全部输入校验 + 规模上限
  sessions/                 # —— 会话存储与增量状态 ——
    accumulator.py          # 滑动窗口 / 指数遗忘 增量协方差 + 批处理参考定义
    store.py                # 原子落盘（config/state/results.jsonl/batches/buffer.npz）
    events.py               # 内存发布订阅（SSE 推送总线）
    session.py              # 会话：并发锁、批次幂等、追加/估计/持久化/重启续写
  api/
    schemas.py              # Pydantic 请求/响应模型
    main.py                 # FastAPI：路由、SSE、错误处理、合成端点
experiments/
  tracking_experiment.py    # 滑动窗口 vs 遗忘因子 阶跃跟踪实验
  results/                  # 实验输出（文档取舍数据）
tests/                      # pytest，逐条覆盖验收关系（统计用例固定种子）
Dockerfile                  # python:3.11-slim + uvicorn
docker-compose.yml          # 带 /data 挂载
```

---

## 2. 快速开始

```bash
# 本地
pip install -r requirements.txt
uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# 或容器
docker compose up --build
# 会话数据落盘在命名卷 doa-data（挂载到容器 /data）

# 测试
pytest
```

打开 `http://localhost:8000/docs` 有交互式 API 文档。环境变量：
`DOA_DATA_DIR`（默认 `/data`）、`DOA_WINDOW_SIZE`（默认 500）、
`DOA_FORGETTING_FACTOR`（默认 0.98）。

---

## 3. API 概览

| 方法 路径 | 说明 |
| --- | --- |
| `POST /sessions` | 建会话：阵列参数 + 估计配置 + 增量更新方式 |
| `GET  /sessions` / `GET /sessions/{id}` | 列表 / 状态（版本、活跃快拍数、已收批次数） |
| `POST /sessions/{id}/append` | 追加一批（实部/虚部分开），带 `batch_id` 幂等；可 `estimate:true` 追加后立即估计 |
| `POST /sessions/{id}/estimate` | 用当前累计协方差出一次估计（可临时覆盖角度范围/步长/平滑/源数/真值角） |
| `GET  /sessions/{id}/results?since_version=N` | **带版本号轮询**：只返回严格新于 N 的结果 |
| `GET  /sessions/{id}/results/latest` | 最新一条 |
| `GET  /sessions/{id}/stream?after_version=N` | **SSE 订阅推送**：先补发历史，再推新结果 |
| `POST /synthesis` | 闭环自测：给各源方位/功率、SNR、快拍数、种子，造数据（可选立即估计/回传数据） |

### 典型流程

```bash
SID=$(curl -s localhost:8000/sessions -H 'content-type: application/json' -d '{
  "array": {"elements": 8, "spacing": 0.5, "wavelength": 1.0},
  "estimation": {"angle_min": -90, "angle_max": 90, "angle_step": 0.5},
  "update": {"mode": "sliding", "window_size": 500}
}' | python -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')

# 实部虚部分开、按批追加（batch_id 重复提交只计一次）
curl -s localhost:8000/sessions/$SID/append -H 'content-type: application/json' -d @batch1.json
# 轮询
curl -s "localhost:8000/sessions/$SID/results?since_version=0"
# 订阅（SSE）
curl -N "localhost:8000/sessions/$SID/stream"
```

每次估计结果包含：`angles`、`mdl_source_count`/`aic_source_count` 及
完整准则曲线、`eigenvalues`、`resolved`、粗扫 `spectrum` 与栅格、
细化容差与区间、`crb_std_deg`、`error_over_crb`（给了真值角时）、
信号/噪声功率估计、有效快拍数、平滑信息、`grating_ambiguity` 等。

---

## 4. 测向内核约定与实现要点

* **方向向量**：`a(θ)[m] = exp(j·2π·p_m·sinθ/λ)`，位置以阵列中心为
  原点。因此 θ→−θ 等价于 a→a*，全局共轭快拍得到反向估计。
* **协方差**：`R = Σ x_k x_k^H / K`。估计前强制厄米对称化。
* **源数判定**：对噪声特征值取几何/算术平均之比的似然项，
  MDL 惩罚 `½ p(k) log K`，AIC 惩罚 `2 p(k)`，`p(k)=k(2M−k)+1`；
  取使准则最小的 k。不显式给源数时默认采用 **MDL**（同时回传 AIC）。
* **MUSIC**：取最小 M−d 个特征向量为噪声子空间，谱
  `1/(a^H E_n E_n^H a)`；先按步长粗扫、取前 d 个局部峰，再在峰两侧
  ±一个步长的区间内做**黄金分割细化，容差 step/100（比步长细两个
  数量级，满足“至少细一个数量级”）**。
* **前后向空间平滑**：子阵长度 L 可配（缺省取使
  `2(M−L+1) ≥ L−1` 成立的最大 L）。前向 `R_f=(1/J)Σ 主子阵`，
  后向 `R_b=J R_f* J`，合成 `(R_f+R_b)/2`。只需要完整阵列 R
  （切子阵=取主子阵），所以与流式累加天然兼容。**可分辨上限**
  `d ≤ min(L−1, 2J)`，超出直接拒绝并在错误详情里给出上限。
  存在阵元位置误差时平移不变性被破坏，禁用平滑（会明确报错）。
* **CRB**：单源、白噪声、确定性（条件）模型
  `var(θ)=½·(σ²/(N·P))/Σ_m(w_m−w̄)²`，`w_m=2π p_m cosθ/λ`，
  返回标准差（度），并在给真值时给出 `|θ̂−θ|/CRB`。在线噪声方差由
  噪声子空间特征值均值给出，并按有效快拍/自由度做一阶无偏修正，
  避免小样本把 CRB 系统性压低。
* **合成数据**：`X=A s+n`，非相干源各自独立 CN(0,P)，相干源共用
  同一复高斯波形；`σ²=(ΣP)·10^(−SNR/10)`。随机数为
  `Generator(PCG64(seed))` 且抽样顺序固定（先信号后噪声），
  **同种子同参数生成的数据逐位相同**（测试断言 `array_equal`）。
* **相位/幅度不变性**：全部快拍同乘任意非零复数 c，R→|c|²R，
  子空间与 MUSIC 角度不变（测试容差 1e-12）。

---

## 5. 增量协方差：两种更新方式

两者都只维护未归一化量，**定义都与“对同一批快拍按同一加权一次
算出”严格一致**，测试要求相对误差 1e-9 量级，实测 1e-15 以下。

### 5.1 定长滑动窗口（**本版默认**，`mode="sliding"`, `window_size=500`）

`R = (1/W) Σ_{k∈window} x_k x_k^H`。维护一个 W 行环形容器和未归一
化和 S。发生驱逐的那一批追加完后，直接由窗口内现存快拍
**重建 S**（`S=B^T conj(B)`，B 为 (W,M) 行存快拍）：

* 重建结果只取决于“窗口内这批快拍的集合”，与到达顺序无关，因此
  **无论追加多久都不会累积扣减误差**；长序列测试（6000 拍、W=256）
  相对批处理定义的误差在 1e-15 量级，且严格保持厄米对称；
* 旧数据在离开窗口瞬间被**干净地剔除**，阶跃后跟踪延迟有上界
  （约一个窗口），这对“测向”这种需要明确丢弃旧朝向的任务更可控；
* 代价是保留窗口内快拍，内存 O(WM)（W≤单批上限量级，M≤64，
  缓冲以 npz 二进制落盘）。

### 5.2 指数遗忘（`mode="forgetting"`, `forgetting_factor=λ`）

对已接收的 N 条快拍取截断几何加权样本均值，逐条递推：

```
S ← λS + x x^H,   w ← λw + 1,   R = S / w
```

由归纳可知最老快拍权重恰为 λ^(N−1)，权重和
`w=1+λ+…+λ^(N−1)`，与批处理参考
（`R=(Σ λ^(N−i)x_i x_i^H)/(Σ λ^(N−i))`）**逐位一致**（切批与整批
送入的差仅在 1e-17 量级）。不需要存历史快拍、内存 O(M²)；但旧数据
按指数**拖尾**衰减、永不被彻底剔除。

### 5.3 默认选择与实验论证

常驻测向更看重“源的朝向变了之后要能明确、有界地甩掉旧数据”，
因此**默认选滑动窗口**（W=500）。遗忘因子方式同样完整支持，适合
内存受限或希望时间常数可调（τ≈1/(1−λ)）的跟踪场景。

用 `experiments/tracking_experiment.py` 做了一次角度阶跃实验
（M=8、半波距、SNR=15 dB，源在第 1200 拍由 **−30° 阶跃到 12°**，
每拍增量更新并出 MUSIC；固定种子 2026；数据在
`experiments/results/step_tracking_15db.json`）：

| 增量方式 | 跟上阶跃所需快拍数¹ | 稳态 RMSE（度）² |
| --- | ---: | ---: |
| 滑动窗口 W=100 | **48** | 0.0254 |
| 滑动窗口 W=300 | 148 | **0.0087** |
| 遗忘 λ=0.95（τ≈20） | **15** | 0.0486 |
| 遗忘 λ=0.98（τ≈50） | 35 | 0.0254 |

¹ 准则：阶跃后估计进入新方向 ±0.5° 走廊且连续 20 拍不再离开。
² 阶跃 600 拍之后全部估计点的均方根误差。

结论与理论一致：**遗忘因子跟踪更快（15–35 拍）但稳态方差更大；
滑动窗口靠加大 W 把稳态 RMSE 压到最低（W=300 时 0.0087°），代价是
延迟与 W 同阶**。同一机制的参数在“延迟—稳态精度”上是对偶的：
要快就用小窗口或小 λ，要稳就用大窗口或接近 1 的 λ。重跑：

```bash
python -m experiments.tracking_experiment --snr 15 --k-per-side 1200
```

---

## 6. 会话、持久化与并发

* **持久化布局**（`DOA_DATA_DIR/<session_id>/`）：
  `config.json`（配置，创建后只读）、`state.json`（累加器元数据+
  最新结果+版本号，原子替换）、`results.jsonl`（每次估计一行，追加）、
  `batches.log`（已计入批次号，幂等依据）、`buffer.npz`（滑动窗口
  环形缓冲，二进制原子写）。所有写盘走临时文件 + fsync + rename。
* **重启续写**：新进程按 id 懒加载会话，恢复累加器状态与已见批次号，
  历史结果不丢，版本号延续，可继续追加（见
  `test_persistence_survives_restart`、
  `test_persistence_forgetting_state_roundtrip`）。
* **多会话隔离**：各会话独立配置、锁、累加器与结果目录。
* **并发与幂等**：每会话一把可重入锁，同一会话的追加/估计串行化。
  两批同时到达时最终状态**等价于按某个先后顺序串行追加**，不丢批、
  不重复；`batch_id` 在锁内检查并落盘，**同一批次重复提交整批跳过**
  （响应里 `duplicate:true`）。并发 12 批的测试同时核对活跃快拍数、
  批次数以及与串行参考会话的特征值一致（1e-10）。
* **轮询/推送**：轮询只返回 `version > since_version` 的结果；
  SSE 先按 `after_version` 补发历史再实时推送，慢消费者只丢自己
  队列里最旧的待发结果，不阻塞追加路径。

---

## 7. 校验、规模上限与错误

统一返回 `422`，结构为
`{"error": {"type": ..., "message": "<中文人读说明>", "details": {...}}}`。

* 规模上限：**阵元 ≤ 64、单批快拍 ≤ 10000、扫描点 ≤ 200000**，
  超限返回 `limit_exceeded`。
* 人读错误：信源数不小于（有效）阵元数；首次估计时累计快拍少于
  阵元数；波长或间距非正；数据含 NaN/Inf；实部虚部形状不一致；
  平滑可分辨上限；扫描范围/步长非法等。
* **栅瓣模糊**：间距 > λ/2 时 `grating_ambiguity.present=true`，
  并列出扫描范围内会被混淆的角度对（满足
  `sinθ′−sinθ = k·λ/d`），供上位机判读。

---

## 8. 验收关系与测试对照（`tests/`，统计用例固定种子）

| 验收关系 | 用例 |
| --- | --- |
| 高 SNR/快拍够多、拉开的两非相干源偏差 < 0.1° | `test_two_incoherent_sources_bias_under_0p1` |
| 全乘非零复数角度纹丝不动（1e-12） | `test_global_complex_scaling_leaves_angles_unchanged` |
| 间距 > 半波长报栅瓣并列出混淆角度对 | `test_grating_lobe_ambiguity_reported_with_pairs` / `test_no_grating_when_half_wavelength` |
| 角度统一变号 → 估计统一变号 | `test_angle_sign_flip` |
| 完全相干两源：不平滑分不开、开平滑分开 | `test_coherent_sources_require_smoothing` |
| SNR 0→20 dB RMSE 下降，高 SNR RMSE/CRB ≤ 3 | `test_rmse_decreases_with_snr_and_approaches_crb` |
| 高 SNR MDL 源数正确（多种子） | `test_mdl_source_count_correct_high_snr` |
| 整批一次送入 ≡ 切批追加（估计一致） | `test_whole_batch_equals_chunked_appends`、`test_forgetting_chunked_equals_single_batch` |
| 增量协方差 ≡ 同定义批处理（1e-9）、长序列不漂移、保持厄米 | `test_accumulator.py` 全部 |
| 批次幂等 / 并发等价串行 / 多会话隔离 | `test_duplicate_batch_id_counted_once`、`test_concurrent_appends_equivalent_to_serial`、`test_sessions_isolated` |
| 重启续写、结果不丢、遗忘状态往返 | `test_persistence_*` |
| 轮询版本过滤 / SSE 历史补发与实时推送 | `test_append_estimate_poll_flow`、`test_sse_*` |
| 合成同种子逐位一致 | `test_synthesis_bit_identical_same_seed`、`test_synthesis_endpoint_bit_identical_then_data_roundtrip` |
| 细化比步长细一个数量级以上 | `test_refinement_finer_than_step_by_order_of_magnitude` |
| 各类人读错误与规模上限 | `test_error_*` / `test_http_*` |
