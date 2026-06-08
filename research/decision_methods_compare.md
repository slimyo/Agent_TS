# 决策方法综合对比（Decision Methods Comparison）

> 版本：2026-06-04。汇总本项目从 Round 5→12 出现过的全部"何时偏离 base 模型"的决策方法，
> 横向对比其决策依据、避险/获利机制、评测口径、实测 vs_base、偏离行为与泄漏风险。
> 配套：`method1-7.md` / `finish1-7.md` / `test/signal_router.py` / `plan.md §零`。

---

## 0. 决策问题定义（所有方法共享）

少样本（N=3/5/10 per class）下，强 base 模型（分类 = Rocket）已接近饱和。
给定一个候选库（5→10 分类器），决策器要回答：**这一格(dataset×N×seed)该守 base，还是偏离到某候选？**

- **目标**：max 准确率，同时 **不显著低于 base**（损害可控）。
- **诚实底线**：全 **LODO**（leave-one-dataset-out），任何信号只用其它域历史 outcome 训练，无 test 泄漏。
- **理论天花板**：oracle（每格选事后最优）。分类饱和域 oracle 头寸仅 +4~4.5pp。

---

## 1. 方法谱系（按演进顺序）

| 代号 | 决策依据 | 避险机制 | 获利机制 | 泄漏风险 |
|---|---|---|---|---|
| **B0 commit-base** | 永远守 base | 天然不偏离 | 无 | 无 |
| **v1 CV-argmax** | 选交叉验证 acc 最高的候选 | 无（盲选 CV 冠军）| CV 冠军 | 无，但 CV 少样本噪声大 |
| **v2 N-cond fallback** | CV-argmax + `N<7→强制 base` | N 门（少样本不信 CV）| CV 冠军 | 无 |
| **v3 memory** | CV + 跨数据集 winner 记忆 | 记忆先验 | 记忆 winner | ⚠️ **有泄漏**（记忆含被测域）|
| **v3 honest** | v3 去泄漏（记忆只用其它域）| 记忆先验 | 记忆 winner | 无 |
| **v3 calibrated** | v3-honest + CV→test isotonic 校准 | 校准后阈值 | 校准 CV | 无（但校准**负结果** [[finding_cv_calibration_negative]]）|
| **deployed RouterAgent** | CV margin + **trust 门**（CV饱和惩罚×fold稳定）| trust 门（避险，F-R11.9）| CV margin | 无 |
| **v2 signal_router（当前）** | **双信号解耦**：conformal_safe **AND** saturation_headroom | **conformal**（AUC0.79）| **saturation**（AUC0.71）| 无 |

> 核心演进逻辑：**v1→v2→v3 是"让 CV 路由更保守"的修补**（盲选 CV → 加 N 门 → 加记忆/校准）；
> 到 deployed/RouterAgent 才把"避险"独立成 **trust 门**（F-R11.7：trust≈P(safe)≠gain）；
> v2 signal_router 是终态——认识到**"避险"和"获利"是两个需不同信号的子问题**（F-R12.5），
> 用 conformal 管避险、saturation 管获利，二者 **AND**，而非求一个万能 confidence/trust。

---

## 2. 实测对比 A：早期 5-数据集域（CV 路由系列）

> 评测域：BeetleFly/BirdChicken/Coffee/ECG200/TwoLeadECG × 3N × 2seed = 30 cells，5-clf 库。
> base = Rocket 87.53%。

| 方法 | acc% | vs base | 评注 |
|---|---|---|---|
| v1 CV-argmax | 84.76 | **−2.77** | 盲信 CV → 少样本噪声，重伤 base |
| v2 N-cond fallback | 86.66 | −0.87 | N 门救回 ~2pp，仍负 |
| **v3 memory** | 88.42 | **+0.89** | ⚠️ 唯一正值，但**含泄漏**，不可信 |
| v3 honest | 86.91 | −0.62 | 去泄漏后正收益消失 → 印证"per-cell winner 难学" |
| v3 calibrated | 86.66 | −0.87 | CV→test 校准**反而更差**（[[finding_cv_calibration_negative]]）|
| base (rocket) | 87.53 | +0.00 | — |

**读法**：① CV 路由全系（去泄漏后）**无一胜过 base**；② v3_memory 的 +0.89 是泄漏假象，
honest 版立刻回落到 −0.62 → **这是"per-cell gain 不可离线可靠学"的最早证据**；③ 校准救不了。

---

## 3. 实测对比 B：全量域（v2 signal_router，当前）

> 评测域：**UCR 24（含新增 FordA/FordB）+ UEA 14 多变量 = 38 datasets / 221 cells**，
> **10-clf 全量库**（UEA 经 channel-flatten 补 7 个），full LODO。base 80.00%。

| 决策 | system | base | vs base | 偏离 |
|---|---|---|---|---|
| **双信号门 (trust≥.5 AND head≥.5)** | **80.00%** | 80.00% | **+0.00pp** | 0/221 |
| ├ UCR 单变量(24) | 84.35% | 84.35% | +0.00pp | 0 |
| └ UEA 多变量(14) | 72.49% | 72.49% | +0.00pp | 0 |
| 自由偏离 (head≥0) | 79.71% | 80.00% | −0.30pp | 221 (safe 0.52) |
| oracle 上界 | 84.46% | — | +4.46pp | — |

**读法**：① 库越大、域越杂（加入多变量、base 更弱 72%），headroom 信号判定**无任何 cell 值得偏离**
→ 精确 **+0.00pp / 0 偏离**，UCR/UEA 两域皆然；② 自由偏离 −0.30pp、safe-rate 0.52（≈掷硬币）
→ 再证 per-cell gain 不可学；③ oracle 4.46pp 逐-cell 不可达。

---

## 4. 研究侧机制证据（为什么终态是双信号解耦）

| Finding | 一句话 | 决定了什么 |
|---|---|---|
| F-R11.7 | trust ≈ P(action safe) ≠ gain ≈ E[Δreward]——是风险估计器非效用估计器 | 避险 ≠ 获利，不能合并 |
| F-R12.1/12.2 | gain 幅度弱可回归(corr 0.40)但"是否获利"分类失败(AUC 0.47)；预测域 17% 头寸也吃不到 | per-cell gain **不可学**（跨饱和/非饱和、分类/预测一致）|
| **F-R12.5** | **避险↔conformal(AUC0.79)、获利↔saturation(AUC0.71)，不同信号；朴素融合稀释** | **决策升级为双信号解耦** |
| F-R12.4 | 轻量 proposal net ≈ LLM（profits 0.235 vs 0.267）| proposal 角色可无-API 平替 |
| F-R12.3 / 12.6 | 组合输单 base；探索离线净负 | portfolio/explore 当前否证 |

---

## 5. 综合结论

1. **CV 路由系（v1→v3）本质是"修补盲选"**：每一步（N 门→记忆→校准）都在降损害，
   但去泄漏后**无一能正向超越 base**——证明"逐-cell 选赢家"在饱和域不可离线可靠完成。
2. **v3_memory 的 +0.89 是泄漏教训**：任何"看起来赢 base"的离线路由都要先查泄漏（honest 版即回落）。
3. **trust 门（deployed）解决了避险**：trust≈P(safe) 能挡住"会变差"的偏离，但它**不预测获利**。
4. **v2 signal_router 是当前终态**：把决策拆成 `conformal_safe AND saturation_headroom`——
   - 用对的信号干对的事（conformal 避险 / saturation 排获利优先级），不再求万能 confidence；
   - 在**全量库 / 全量数据 / 跨单变量-多变量域**上稳定收敛到 **+0.00pp、0 损害、可解释**；
   - 这不是"没找到更好模型的失败"，而是**饱和律下的正确决策**：当 base 已足够，最优动作就是诚实守 base，
     系统的价值在于**风险可控的选择性弃权**（risk-controlled selective abstention）。

**主线收敛**：CV-argmax盲选 → N门/记忆/校准修补 → trust门只避险 → **双信号解耦（避险↔conformal / 获利↔saturation）**。

---

## 6. 一句话选型建议

> 若 base 模型在目标域已强（典型 TSFM/Rocket 饱和）→ 用 **v2 signal_router 双信号门**，
> 预期 = 诚实守 base、零损害、决策可解释；**不要**用裸 CV-argmax（−2.77pp）或盲信任何"离线赢 base"的路由（查泄漏）。
> 真要拿到正收益，需要的是**未饱和域 / 真互补资产池 / 在线反馈**（见 finish7 §9 future），而非更复杂的离线 confidence。
