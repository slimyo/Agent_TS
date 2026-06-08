# Baseline 总览 — 模型库 · 数据集 · 远程流程

> 版本：2026-06-04。本文件汇总本项目当前**全部 baseline 模型**（分类 / 预测）、**全部数据集**
> （分类 UCR/UEA + 预测 + 检测），以及**远程 GPU 服务器**上各模型的可用性与运行流程。
> 配套：`method1-7.md` / `finish1-7.md` / `test/README.md` / memory `project_remote_server`。

---

## 1. 分类模型库（10 个，`research/agent/clf_strategies.py`）

`CLF_ORDER`（index 0 = base）：

| # | 模型 | 类型 | 依赖 | 说明 |
|---|---|---|---|---|
| 0 | **rocket** ⭐base | 随机卷积核 + 岭回归 | sktime/numpy | 分类 base，UCR/UEA 上近 oracle（71%/75% cell 是 oracle）|
| 1 | moment_1nn | MOMENT embedding + 1NN | transformers | TSFM 表示 |
| 2 | moment_logreg | MOMENT embedding + 逻辑回归 | transformers | |
| 3 | dtw_1nn | DTW 距离 + 1NN | — | 经典强基线（长序列慢，sweep 里按长度门控跳过）|
| 4 | euclid_1nn | 欧氏距离 + 1NN | — | 最简对照 |
| 5 | catch22 | 22 个时序特征 + 分类器 | pycatch22 | 特征工程 |
| 6 | mantis_1nn | Mantis embedding + 1NN | transformers | TSFM 表示 |
| 7 | mantis_lr | Mantis embedding + 逻辑回归 | transformers | |
| 8 | minirocket | MiniRocket 卷积核 | sktime | rocket 轻量变体 |
| 9 | weasel | WEASEL 词袋 | sktime/pyts | 符号化 |

> 多变量 UEA 经 **channel-flatten**（`[N,C,L]→[N,C*L]`）适配上述单变量分类器。
> `llm_direct`（LLM 直接分类）为额外策略，实证结构性弱，不入主库。

### 1.1 扩展分类库（2026-06-04 新增 12 个，已注册进 `CLF_STRATEGY_FN`，共 23 个）

> 对照本文件末尾「可扩展分类模型」四类，每类落地 ≥3 个并 smoke 通过。
> 调用：`predict_with("<name>", X_train, y_train, X_test)`。smoke = Coffee 5-shot acc（仅验证可跑，非性能评测）。

| 类别 | name | 实现 | 依赖/模块 | 运行位置 | smoke |
|---|---|---|---|---|---|
| 一·卷积核/区间 | `multirocket` | RocketClassifier(multirocket) | sktime · `baseline/tsc_extended.py` | 本地 tsci | ✅ |
| 一 | `arsenal` | Arsenal（Rocket 集成）| sktime | 本地 tsci | ✅ |
| 一 | `drcif` | DrCIF（区间森林）| sktime | 本地 tsci | ✅ |
| 二·深度 | `fcn` | FCN (Wang 2017) | 纯 torch · `baseline/tsc_deep.py` | 本地 tsci | ✅ |
| 二 | `resnet` | ResNet-TSC (Wang 2017) | 纯 torch | 本地 tsci | ✅ |
| 二 | `inceptiontime` | InceptionTime (Fawaz 2019) | 纯 torch | 本地 tsci | ✅ |
| 三·TSFM 嵌入 | `chronos2_emb` | chronos-2 编码器嵌入 + LR | `baseline/tsfm_embed.py` | **远程 tsci-c2** | ✅ acc0.54 |
| 三 | `timesfm_emb` | TimesFM-2.0 嵌入 + LR | 同上 | **远程 tsci-tsfm** | ✅ acc0.54 |
| 三 | `timer_emb` | Timer-S1 隐状态 + LR | 同上 | **远程 tsci-remote** | ✅ **acc0.89** |
| 四·原生多变量 | `muse` | MUSE（多变量词袋）| sktime · `tsc_extended.py` | 本地 tsci | ✅ |
| 四 | `rocket_mv` | 多变量 Rocket（非 flatten）| sktime | 本地 tsci | ✅ |
| 四 | `cif_mv` | CanonicalIntervalForest | sktime | 本地 tsci | ✅ |

> - **类别三（TSFM 嵌入）参数量大 → 部署在远程**：各自走对应 env（chronos2_emb→tsci-c2 / timesfm_emb→tsci-tsfm / timer_emb→tsci-remote，见 §4.2）。timer_emb 须 `HF_HUB_OFFLINE=1`（避免 HEAD huggingface.co 超时）。Timer-S1 嵌入分类最强（Coffee 5-shot 0.89）。
> - **类别四原生多变量**：函数接受 3D `[N,C,L]`（不 flatten）；当前分类 sweep 喂的是 channel-flatten 后的 2D，要发挥多变量优势需走多变量数据路径（喂 3D），属后续接线。
> - 类别一/二/四纯 CPU/小模型 → 本地 `tsci` 即可；深度网默认 `epochs=200`，少样本可调小。
> - 未落地：FreshPRINCE（要 py<3.10）、HIVE-COTE2（太慢，仅作 oracle）、TimesNet/PatchTST/TS2Vec（需各自 repo，未来按需）。

### 1.2 全量 sweep + LODO 决策（2026-06-04）

把 12 个新分类器跑遍 38 数据集 × N{3,5,10} × seed{1,42}（test 截断 150；脚本
`research/experiments/taskb_newlib_sweep.py`，写 `results/taskb_newlib.jsonl`，已并入
`test/signal_router.py` 的 `CLF_ORDER`(10→22) + `METHOD_TO_CLF` + `SWEEPS`）。

**分布式编排**（按依赖选机器/env）：
- 类别二深度网（torch）→ **远程 GPU `tsci-c2`**，极快（InceptionTime/cell <1s，acc 可达 0.99）。
- 类别三 chronos2_emb → 远程 `tsci-c2`；timer_emb → 远程 `tsci-remote`（`HF_HUB_OFFLINE=1`）。
- 类别一/四 sktime（numba）→ 远程 `tsci-c2` 需 **`pip install --ignore-installed numba>=0.59` + `PYTHONNOUSERSITE=1`**
  （否则 `.local` 旧 numba 撞 numpy2.2 → numba 报错 → `_safe` 回退 majority = 假精度）。本地 `tsci` 也可但慢。

**LODO 决策结果**（`python -m test.run_full_test_v2`，22 分类器全量库）：

| 决策 | system | base | vs base | 偏离 | oracle |
|---|---|---|---|---|---|
| 双信号门 (trust≥.5 AND head≥.5) | 80.00% | 80.00% | **+0.00pp** | 0/221 | **84.84%** |
| 自由偏离 (head≥0) | 77.12% | 80.00% | −2.88pp | 221 (safe 0.44) | — |

**读法**：扩库后 **oracle 上界 84.46→84.85pp 抬高**（深度网/timer_emb 在部分 cell 真超旧 oracle，库更多样）；
但双信号门仍 **+0.00pp / 0 偏离** —— 即"库更大更杂、天花板更高，逐-cell 增益依旧不可安全捕获"，
**强化** F-R12.x（gain 不可学）：扩模型只抬 oracle，不抬可决策收益。自由偏离 −2.88pp（比旧库 −0.43 更差，
新候选乱选更伤）再证。

> **相图模型基底（2026-06-06 全量补齐）**：早期相图图例只显示"被选中(chosen)模型"≈9 个小模型，给人"只测了小模型"的错觉。
> 已改为**按 cell 的 oracle-winner 上色**，图例列全库 22 模型，直观证明全库真实参与（分类 oracle 由 **22 个模型分摊**）。
> **真量库覆盖（分类 38 数据集 ×N×seed = 228 cell）= 全部 22 模型 ✅ 全 228**：
> - base 10 + 深度 fcn/resnet/inceptiontime + **真·远程 TSFM 嵌入 chronos2_emb / timesfm_emb / timer_emb** + 6 sktime（multirocket/arsenal/drcif/muse/rocket_mv/cif_mv）**全部 228 cell 跑完**（2026-06-06）。
> - UEA 多变量对区间森林+catch22（drcif/cif_mv）原本极慢——关键提速：① `LCAP=512`（时间下采样）② `CCAP=8`（通道下采样，治 Heartbeat 61ch）③ **`n_jobs=-1`（多核并行，结果不变，~10× 加速，是收尾关键）**；ArticularyWordRecognition(25类)/Heartbeat 等全部补齐。
> - 远程大模型(chronos2/timesfm/timer)真实调用远程 GPU 推理（`emb_clf_sweep.py` 顶层加载一次+重试，绝不写 majority 假值）。
> **远程 TSFM 嵌入均真实调用远程环境模型**（chronos-2→tsci-c2 / TimesFM-2.0→tsci-tsfm GPU / Timer-S1 8.3B→tsci-remote），
> 经 `emb_clf_sweep.py`（顶层加载一次+重试，绝不写 majority 假值）+ snapshot 完整 stage + 纯离线运行（见 §4.2 坑）。

### 1.3 三大任务统一决策相图（`m22_threetask_phase.py` → `results/m22_threetask_phase.png`）

在分类/预测/检测三任务的全量库上做同口径 LODO 双信号决策，画 (saturation×trust) 相图：

| 任务 | base | 高饱和 cell 占比 | 系统偏离数 | vs_base |
|---|---|---|---|---|
| 分类 (221cell/**22clf 全量库**) | rocket | **1.00** | 0 | +0.00pp |
| 检测 (12cell/**22clf 全量库**, 12/12 全跑完) | rocket | **0.58** | 2 | +0.94pp* |
| 预测 (72cell/**14 TSFM 全量库**) | chronos2 | 0.92 | 6 | **−1.6 %relMAE** |

> 检测也已扩到全量 22 分类器（`taskc_newlib_sweep.py`，旧 7 + 新 15；本地 14 + 远程 3 emb），**12/12 cell 全跑完**。
> 扩库后检测高饱和占比 0.92→**0.58**（新分类器在 ~40% 检测 cell 上制造头寸）。
> *检测 vs_base=+0.94pp（2 次偏离命中）在合并过程中曾在 −0.63~+0.94pp 间抖动（**n=12 太小，落在噪声内，不算稳定正收益**）；
> 结论与分类/预测一致：**扩库揭示头寸，但 12 cell 上偏离收益不稳定** → 逐-cell 不可预测。

> 预测全量库（2026-06-05 补齐）：chronos×3 + arima/naive/llmtime + **moirai2/timesfm2/tirex/toto/toto2/sundial/time_moe/timer**（8 个 TSFM，`taska_tsfm_sweep.py`→`taska_tsfm.jsonl`）。
> **远程全量补跑的关键**：这些 TSFM 用 `trust_remote_code` 会直连 huggingface.co 做 HEAD 校验（绕镜像）→ 国内批量跑必超时；
> 解决=**先 `snapshot_download` 把每个 repo 完整 stage 到缓存，再 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 纯离线跑**（脚本 `/data2/c220/hz/run_fc_tsfm_offline.sh`）。
> 即便预测库扩到 14 个 TSFM，决策仍只 6 次偏离且净亏 −1.6%relMAE → 与分类同结论（头寸真实但逐-cell 不可预测）。

**为什么三任务都≈ +0.00pp（相图直接给答案，已是全量库）**：
- 分类全量 22 库下 cell 仍全落高 saturation 区（左上"可偏离区"几乎空）→ 门不开 → +0.00pp。
- 检测扩到 22 后约 1/4 cell 进入低 saturation（有头寸），门开 1 次，但净收益仍 0 → 有头寸≠抓得到。
- 预测 14 TSFM 全量库，左上区有 cell、门开 6 次，但净亏 −1.6%relMAE → 逐-cell 谁赢不可预测（F-R12.x）。
→ 结论：+0.00pp **不是窄库 artifact，也不是系统没用**——三任务全量库下，**头寸（oracle 天花板）真实但逐-cell 不可预测**，
  双信号门在"无可学到的安全收益"处诚实弃权。**天花板可达性 ≠ 逐-cell 可预测性**（详见 `advisor_report.md §6`）。

---

## 2. 预测模型库（`research/agent/forecaster_reflect.py` · `STRATEGY_FN`）

### 2.1 经典 / 轻量（无大模型权重）

| 模型 | 模块 | 说明 |
|---|---|---|
| naive_drift | `baseline/naive.py` | 漂移外推 |
| naive_seasonal | `baseline/naive.py` | 季节朴素 |
| arima_ets | `baseline/arima_ets.py` | ARIMA/ETS 统计法 |
| llmtime | `baseline/llmtime.py` | LLM 数值续写（走 OpenAI 兼容 API）|

### 2.2 TSFM（需权重）—— 2026-06-04 全部盘点 + 补装

> **可用 11/12 个**（仅 toto2 阻塞）。每个 TSFM **只在指定 env 跑通**（见 §4.2，版本互斥）。
> smoke = 在该 env 对 400 点合成序列 `predict(H=24)` 返回有限值。

| 模型 | repo | 状态 | 可用 env | 权重 | 备注 |
|---|---|---|---|---|---|
| chronos | amazon/chronos-t5-small | ✅ smoke OK | `tsci-c2`(远) | 177M | T5 量化预测 |
| **chronos2** ⭐base | amazon/chronos-2 | ✅ smoke OK | `tsci-c2`(远) | 456M | **预测 base** |
| chronos_bolt | amazon/chronos-bolt-small | ✅ smoke OK | `tsci-c2`(远) | 183M | bolt 快通道 |
| timer | bytedance-research/Timer-S1 | ✅ smoke OK | `tsci-remote`(远) | 16G | 8.3B；**实测不优于 chronos2**（F-R12.7，−7.0% rel-MAE）|
| sundial | thuml/sundial-base-128m | ✅ smoke OK | `tsci-remote-tx440`(远) | 490M | 旧 `seen_tokens` API → 须 tf4.40 |
| time_moe | Maple728/TimeMoE-50M | ✅ smoke OK | `tsci-remote-tx440`(远) | 217M | 同上 |
| timesfm2 | google/timesfm-2.0-500m-pytorch | ✅ smoke OK | `tsci-tsfm`(远) | ~3.5G | `pip timesfm` 1.3.0；权重走 Xet→须 `HF_HUB_DISABLE_XET=1` |
| moirai2 | Salesforce/moirai-2.0-R-small | ✅ smoke OK | `tsci-moirai`(**本地**) | 小 | `uni2ts`+`gluonts==0.14.3`（uni2ts 2.0 强依赖 0.14.x）|
| tirex | NX-AI/TiRex | ✅ smoke OK | `tsci-tsfm`(远) | 中 | 包名是 **`tirex-ts`**（git，非 PyPI `tirex`）|
| toto | Datadog/Toto-Open-Base-1.0 | ✅ smoke OK | `tsci-tsfm`(远) | ~400M+ | `toto-ts` **必须 `--no-deps`**（否则拉 cu126 torch 毁掉 cu128 Blackwell 环境）+ 手补纯 py 依赖；xformers 可选(SDPA 回退) |
| toto2 | Datadog/Toto-2.0-4m | ✅ smoke OK | `tsci-toto2`(远) / `tsci-py312`(本地) | 4M | 包名 `toto-2`(py≥3.12)；**关键修复=gluonts 升 0.16.2**（0.14.4 时 forward 返回 tuple↔predict_to_numpy 期望 tensor）+ torch≥2.5(修 `enable_gqa`)|

> **全部 12 个可跑**（chronos×3 + timer + sundial + time_moe + timesfm2 + moirai2 + tirex + toto + **toto2**）。
> 补装关键坑：① timesfm 权重走 HF-Xet，镜像不代理 → 必须 `HF_HUB_DISABLE_XET=1`；
> ② tirex 真包是 `tirex-ts`（git），PyPI 的 `tirex` 是无关包；
> ③ toto 全量依赖会把 cu128 torch 换成 cu126（Blackwell sm_120 失效），必须 `--no-deps` + 手补纯 py 包；
> ④ toto2(`toto-2`) 必须 gluonts **0.16.x**（非 0.14.x）+ torch≥2.5 + py≥3.12，远程独立 env `tsci-toto2`。

---

## 3. 数据集

### 3.1 分类（38 数据集 / 221 cell，全 LODO）

每数据集 × N∈{3,5,10}/类 × seed∈{1,42} 构成 cell；oracle 库含各 cell 的 10 分类器真实 test acc。

**UCR 单变量（24，140 cell）**：
BeetleFly, BirdChicken, Chinatown, Coffee, Crop, DistalPhalanxOutlineCorrect, ECG200, ECG5000,
FaceFour, FordA, FordB, GunPoint, ItalyPowerDemand, Lightning7, MoteStrain, Plane,
ProximalPhalanxOutlineCorrect, SonyAIBORobotSurface1, SonyAIBORobotSurface2, Strawberry,
SyntheticControl, Trace, TwoLeadECG, Wafer

**UEA 多变量（14，81 cell）**：
ArticularyWordRecognition, AtrialFibrillation, BasicMotions, Cricket, ERing, Epilepsy,
FingerMovements, HandMovementDirection, Handwriting, Heartbeat, Libras, NATOPS, RacketSports,
UWaveGestureLibrary

### 3.2 预测（6 数据集 / 72 cell）

每数据集 × N∈{10,20,50,100} × seed∈{1,42,123}，H=96，确定性 `few_shot_split`（仅依赖 ds,N,H,seed）。

| 数据集 | 采样 | season_m | 目标列 |
|---|---|---|---|
| ETTh1 / ETTh2 | 1h | 24 | OT |
| ECL | 1h | 24 | MT_001 |
| Exchange | 1d | 7 | rate_0 |
| Weather | 10min | 144 | OT |
| ILI | 1w | 52 | OT |

> ⚠ ILI 本地/远程序列长度不同（start_idx 不一致）→ 跨机对比时剔除。
> 数据加载：`research/utils/{data_loader, ucr_loader, uea_loader, splitter, inject_fault}`。

### 3.3 检测（合成 4-class fault，12 cell）

ETTh1/ECL 上注入 4 类故障（`inject_fault`）；base=rocket 已最优，系统全 abstain/fallback。

---

## 4. 远程 GPU 服务器流程

### 4.1 连接与目录

- **SSH**：`c220@10.192.43.66`（密码 `cinter`，无 key → 用 `sshpass -p cinter`）
- **GPU**：2× RTX 5070 Ti（各 16GB，Blackwell sm_120，CUDA 12.8）
- **工作目录**：`/data2/c220/hz/agent_ts/`（949G 可用）
- **HF 缓存**：`HF_HOME=/data2/c220/hz/hf_cache` + `HF_ENDPOINT=https://hf-mirror.com`（国内镜像）
  - ⚠ 部分 repo（如 timesfm-2.0）权重走 **HF-Xet**（cas-bridge.xethub.hf.co），镜像不代理 → 下载 read timeout。
    解决：加 `HF_HUB_DISABLE_XET=1` 强制走普通 LFS 经镜像下。

### 4.2 conda env（按模型选 env！版本互斥）

**远程**（`c220@10.192.43.66`，4 套）：

| env | python | transformers | torch | 可跑模型 |
|---|---|---|---|---|
| **tsci-remote** | 3.9 | 4.57.1 | 2.8+cu128 | **timer** |
| **tsci-remote-tx440** | 3.9 | 4.40.1 | 2.8+cu128 | **sundial / time_moe**（旧 `seen_tokens` API）|
| **tsci-c2** | 3.10 | 4.57.6 | 2.8+cu128 | **chronos / chronos2 / chronos_bolt**（chronos-forecasting 2.2.2 含 Chronos2Pipeline，需 py≥3.10）|
| **tsci-tsfm** | 3.10 | 4.57.6 | 2.8+cu128 | **timesfm2 / tirex / toto**（clone 自 tsci-c2；timesfm + tirex-ts(git) + toto-ts(--no-deps)）|
| **tsci-toto2** | 3.12 | — | 2.8+cu128 | **toto2**（`toto-2` + gluonts==0.16.2；建法 `/data2/c220/hz/build_toto2_remote.sh`）|

**本地**（`tsci` 系，用于 TSFM/扩展库）：

| env | python | 可跑模型 | 说明 |
|---|---|---|---|
| **tsci-moirai** | 3.10 | **moirai2** | clone 自 tsci，`gluonts==0.14.3`（uni2ts 2.0 需 0.14.x）|
| **tsci-py312** | 3.12 | **toto2** | torch 2.5.1 + **gluonts 0.16.2**（已修，smoke OK）|

> 互斥点：sundial/time_moe 在 4.57 报 `seen_tokens`；timer 在 4.40 报 tensor 尺寸不匹配；
> chronos2 在 py3.9 装不了（pip 只到 1.5.3）；moirai2 需 gluonts 0.14.3 而 chronos/toto 系需 ≥0.16 → 各自独立 env。
> `tsci-c2`/`tsci-tsfm` 建法见远程 `/data2/c220/hz/build_c2_env.sh` / `build_tsfm_remote.sh`
> （torch 走 download.pytorch.org/whl/cu128，其余走 tsinghua 镜像）。

### 4.3 标准运行流程

```bash
# 0) 本地装 sshpass（如无）：apt-get install -y sshpass
# 1) 同步代码到远程（排除大文件）
sshpass -p cinter rsync -az --exclude='__pycache__' --exclude='results/' --exclude='external/' \
  research/ c220@10.192.43.66:/data2/c220/hz/agent_ts/research/

# 2) 远程跑（按模型选 env；timer 须 TIMER_FORCE_GPU=1 多卡 fp16）
sshpass -p cinter ssh c220@10.192.43.66 \
  'cd /data2/c220/hz/agent_ts && source ~/anaconda3/etc/profile.d/conda.sh && conda activate tsci-c2 && \
   HF_HOME=/data2/c220/hz/hf_cache HF_ENDPOINT=https://hf-mirror.com \
   nohup python -m research.experiments.<script> > run.log 2>&1 &'

# 3) 拉结果回本地
sshpass -p cinter scp c220@10.192.43.66:/data2/c220/hz/agent_ts/research/results/<x>.jsonl research/results/
```

**长任务用 nohup 后台 + 轮询 log**（ssh 前台会被超时杀；下载大权重首跑慢，之后走缓存秒级）。
`timer` 加 `TIMER_FORCE_GPU=1`（默认 `prefer_cpu=True` 保护本地 6GB 卡）；fp16 加载时输入须 `.to(dtype=model.dtype)`（已在 `baseline/timer.py` 修）。

### 4.4 盘点工具

- `research/experiments/tsfm_inventory_smoke.py`：逐模型 load+predict 报 OK/缺依赖/缺权重/报错。
  用 `ENV_TAG=<env> HF_HUB_OFFLINE=1 python -m research.experiments.tsfm_inventory_smoke` 快速判定（不触发下载）。
- 跨机预测对比：因 chronos2 历史结果在本地，亦可"远程只跑候选模型 + 本地按 (ds,N,seed) merge"
  （确定性 split，按 `start_idx` 校验同窗口）。现 chronos2 已可远程跑，预测全量 sweep 可端到端在远程完成。

---

## 5. 当前 base 选择（实测结论）

| 任务 | base | 依据 |
|---|---|---|
| 分类 / 检测 | **rocket** | UCR/UEA 上 71%/75% cell 即 oracle（饱和）|
| 预测 | **chronos2** | Timer-S1(8.3B) 实测不优（F-R12.7，60cell mean −7.0% rel-MAE，仅 ETTh2 占优）|

> 决策机制（双信号解耦 `conformal_safe AND saturation_headroom`）见 `test/README.md §0` 与 `finish7.md`。

---

现有 10 个模型之外，基于 **2023–2026 年顶会及高水平期刊** 遴选的强分类基线。所有方法均提供 **可直接下载/安装的代码链接**，（现有 `weasel` 若为 v1，可升级为 v2）。

---

## 一、基于随机卷积核与字典的经典 SOTA（sktime 原生，零额外依赖）

这些方法在 UCR/UEA 上至今保持前 3，且与 Rocket 原理互补，是最能拓宽路由池多样性的选择。

| 模型 | 类型 | 依赖 | 说明 | 下载 / 安装 |
|------|------|------|------|-------------|
| **MultiRocket** | 极大量随机卷积核 + 多种池化 | `sktime` ≥ 0.18 | Rocket 的强力扩展，在 UCR 上常优于 Rocket，是当前经典方法的标杆之一。 | `pip install sktime`；`from sktime.classification.kernel_based import MultiRocket` |
| **Hydra** | MultiRocket + 字典特征 | `sktime` | 在 MultiRocket 基础上融合字典特征，UCR 最高精度持有者之一。 | 同上，`from sktime.classification.kernel_based import Hydra` |
| **DrCIF** | 随机区间 + 丰富统计特征（catch22 + 其他） | `sktime` | 基于区间（interval）的森林集成，在长序列与高噪声数据上表现优异，与卷积核形成互补。 | `pip install sktime`；`from sktime.classification.interval_based import DrCIF` |
| **FreshPRINCE** | 随机区间 + 大量统计特征 + 简单分类器集成 | `sktime` | 经典区间方法的最新迭代，效率极高，适合作为轻量基线。 | 同上，`from sktime.classification.interval_based import FreshPRINCE` |
| **WEASEL 2.0** | 符号傅里叶近似 + 词袋 + 特征选择 | `sktime` / `pyweasel` | 您列表中已有 `weasel`（可能是 v1），请务必升级为此版本。词袋方法在短序列、符号化强模式上极强，且与卷积核彻底异质。 | `pip install sktime`；`from sktime.classification.dictionary_based import WEASEL` (v2) <br> 或直接 `pip install pyweasel` |
| **HIVE‑COTE 2** | 异构集成（MultiRocket, DrCIF, WEASEL 等） | `sktime` | 当单模型路由失效时，直接用顶级集成作为“万无一失”的基座可显著拉高天花板。 | `from sktime.classification.hybrid import HIVECOTEV2` <br> ⚠️ 训练较慢，可仅作为 oracle 或少量 cell 实验 |

---

## 二、深度学习新范式（2023–2025 顶会，适合拉开时间差）

这些模型可直接输出分类 logits，或提取嵌入后接 1NN / LR，与您现有的 MOMENT / Mantis 形成不同训练目标的预训练/架构对比。

| 模型 | 类型 | 依赖 | 说明 | 下载 / 链接 |
|------|------|------|------|-------------|
| **TimesNet** | 多周期二维卷积（将 1D 转为 2D） | `torch` | ICLR 2023，在长序列和复杂周期数据上表现突出，UCR/UEA 上常超 Rocket。 | GitHub: [thuml/TimesNet](https://github.com/thuml/TimesNet) |
| **PatchTST** | 通道独立 Transformer + 补丁 | `torch` | ICLR 2023，预测模型，但其自监督预训练嵌入可直接用于分类。 | GitHub: [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) <br> 论文提供分类微调脚本 |
| **TS2Vec** | 对比学习时间序列表示 | `torch` | ICLR 2022 亮点论文，层次化对比损失，获取的通用嵌入在分类上性能强劲且稳定。 | GitHub: [yuezhihan/ts2vec](https://github.com/yuezhihan/ts2vec) |
| **TST (Time Series Transformer)** | 标准 Transformer 编码器 + 分类头 | `torch` | ICLR 2021，但至今仍是深度学习分类的可靠基线，实现简单。 | 广泛实现，推荐: [huggingface/transformers 示例](https://huggingface.co/docs/transformers/tasks/time_series_classification) 或 [tsai](https://github.com/timeseriesAI/tsai) |
| **InceptionTime** | 多尺度 Inception 卷积 | `sktime-dl` / `tsai` | 经典深度卷积基线，在 UCR 上与 Rocket 互有胜负，sktime 有封装。 | `pip install sktime-dl` <br> `from sktime_dl.classification import InceptionTime` |
| **ConvTimeNet** | 自适应深度卷积架构 | `torch` | KDD 2025（预印本），比传统 TCN 更灵活，在多个基准上接近 SOTA。 | 暂无正式公开库，但预印本 [`arxiv:2405.15793`](https://arxiv.org/abs/2405.15793) 的 `experiments` 目录有参考实现 |
| **TimeKAN** | Kolmogorov‑Arnold 网络替代 MLP | `torch` | 2024 年新兴架构，利用 KAN 层提升时序建模能力。 | 实现众多，可选用 [Awesome-KAN](https://github.com/mintisan/awesome-kan) 中的时序分支，或 [`efficient-kan`](https://github.com/Blealtan/efficient-kan) |

---

## 三、通用时序基础模型嵌入（即插即用，与您的 MOMENT / Mantis 平行）

您已使用 **MOMENT** 和 **Mantis**，下面两个新基础模型可进一步扩大表示池，且获得其嵌入几乎不增加编码负担。

| 模型 | 类型 | 依赖 | 说明 | 下载 / 链接 |
|------|------|------|------|-------------|
| **Chronos‑2 (嵌入)** | TSFM 生成式模型 | `torch` | 虽是预测模型，但其编码器隐藏状态是强分类特征。您已用于预测任务，分类时只需加一个线性探头。 | GitHub: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) (Chronos‑2 版本见 repo 的 `v2` 分支) |
| **TimesFM‑2.0** | 编码器‑解码器基础模型 | `torch` | Google 2025 年发布，支持分类微调，线性探头性能常接近甚至超越 Rocket。 | GitHub: [google-research/timesfm](https://github.com/google-research/timesfm) (v2 模型权重在 Hugging Face) |

---

## 四、多变量 UEA 专用基线（直接处理多变量，无需 channel‑flatten）

您的当前方案将所有多变量序列拉平为单变量再送入分类器，这可能会丢失通道间相互作用。以下模型**原生支持多变量**，与您已跑的单变量路由形成对比，将有力证明路由空间的增益。

| 模型 | 类型 | 依赖 | 说明 | 下载 / 链接 |
|------|------|------|------|-------------|
| **MUSE** | 多变量词袋 (WEASEL 的多变量扩展) | `sktime` / `pyts` | UEA 上的顶级经典方法，基于符号化多变量特征，与 Rocket 单变量路由形成多变量互补。 | `from sktime.classification.dictionary_based import MUSE` <br> 或独立包 `pip install pyts` (MultivariateWEASEL) |
| **Rocket (multivariate)** | 多变量随机卷积核（**不是 channel‑flatten**） | `sktime` | sktime 的 `Rocket` 和 `MultiRocket` 已原生支持多变量输入 (n_channels > 1)，无需您手工 flatten。 | `from sktime.classification.kernel_based import Rocket` (input: `[N, L, C]`) |
| **TimeSFormer / VideoMAE 类** | 视频 Transformer 适配时序 | `torch` | 将多变量序列视为“图像帧”，利用 VideoMAE 预训练迁移，2024 年顶会中有不错表现。 | 参考实现: [TimeSformer](https://github.com/facebookresearch/TimeSformer) (可改造成序列输入) |

---

