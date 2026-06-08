"""Trust-Aware Online Router —— 工业版选择性行动 agent（解 feedbackm7 三大缺陷）。

feedbackm7 诊断：method7 强在 L1(避险)、弱在 L2/L3(收益)，三大工业缺陷=
  ① 完全离线无反馈  ② 无状态/episode  ③ 资产池同质。
并指出关键纠正：**"gain is not learnable under OFFLINE features" ≠ "fundamentally unlearnable"**——
带反馈的 Contextual Bandit 能学到离线学不到的 winner。

本模块据此把"一次性离线路由"升级为**带反馈的在线 bandit**：
  choose model → observe reward → update posterior  （持续闭环）

三缺陷的对策：
  ① 在线反馈：Thompson 采样 bandit，部署后观测真实 reward(realized acc−base) 更新后验。
  ② 状态/episode：后验**按 domain(dataset) 持久化**——同域重复服务时累积反馈（路线3 episode routing）。
  ③ 资产池/能力画像：每个 model 的后验用 **capability profile** 暖启动——
     = 该 model 在"其它域历史"上的平均 reward（LODO，无泄漏，路线2 task↔model 匹配的轻量版）。
  L1 保留：conformal/保守护栏挡灾难性探索（reward 明显为负的候选不再反复试）。

诚实：capability 先验只用**其它域**；在线更新用**本域部署后观测到的真实 reward**（= 工业反馈，合法）。
"""
from __future__ import annotations

import numpy as np


class TrustAwareOnlineRouter:
    """Per-domain Thompson-sampling bandit over {base + candidates}，capability 先验暖启动。"""

    def __init__(self, models, base, cap_prior=None, prior_strength=2.0,
                 obs_noise=0.05, safety_margin=0.03, discount=1.0, seed=0):
        self.models = list(models)
        self.base = base
        self.rng = np.random.default_rng(seed)
        # capability profile 先验：{model: 期望 reward(acc-base) on 其它域}；缺省 0
        self.cap = cap_prior or {m: 0.0 for m in self.models}
        self.k0 = prior_strength            # 先验伪计数（越大越信先验，episode 短时关键）
        self.obs_noise = obs_noise          # 观测噪声标准差
        self.margin = safety_margin         # L1 安全边际：明显负收益候选退场
        self.gamma = discount               # 遗忘因子 γ∈(0,1]：<1 时旧反馈指数衰减 → 抗漂移
        # per-domain 后验状态：{domain: {model: [n, sum_reward, sumsq]}}（n/sum 为折扣有效计数）
        self.state = {}

    def _post(self, domain, m):
        """返回该 (domain,model) reward 的后验 (mean, std)。先验 = capability。"""
        n, s, _ = self.state.setdefault(domain, {}).get(m, (0.0, 0.0, 0.0))
        mu0 = self.cap.get(m, 0.0)
        # Normal-Normal 共轭近似：后验均值 = (k0*mu0 + n*xbar)/(k0+n)
        xbar = (s / n) if n > 0 else 0.0
        post_mean = (self.k0 * mu0 + n * xbar) / (self.k0 + n)
        post_var = (self.obs_noise ** 2) / (self.k0 + n)   # 不确定性随观测下降
        return post_mean, np.sqrt(post_var)

    def act(self, domain, candidates, topk=5):
        """Thompson 采样选 model。base 是锚点：reward≡0 但带"该不该冒险"的不确定性。
        工业化要点：
          - **候选 shortlist**：只在 capability 先验 top-k 候选里探索（不浪费部署在 22 个模型上）。
          - **base 锚定**：候选要明显采样 >0 才偏离；已多次观测且明显更差的候选退场（L1）。"""
        # shortlist：按 capability 先验取 top-k 非 base 候选
        cand = [m for m in candidates if m != self.base]
        cand = sorted(cand, key=lambda m: self.cap.get(m, 0.0), reverse=True)[:topk]
        best_m, best_s = self.base, 0.0   # base 锚点 reward=0
        for m in cand:
            mu, sd = self._post(domain, m)
            n = self.state.get(domain, {}).get(m, (0,))[0]
            if mu < -self.margin and n >= 2:
                continue                  # L1：已学到明显更差 → 退场
            s = self.rng.normal(mu, max(sd, 1e-6))
            if s > best_s:                # 仅当采样 reward 超 base(0) 才考虑偏离
                best_m, best_s = m, s
        return best_m

    def update(self, domain, m, reward):
        """部署后观测真实 reward，更新后验（在线反馈闭环）。
        γ<1 时**只对被选中模型**的旧统计折扣 → 该模型的新反馈权重更高，可在漂移后快速翻转。"""
        if m == self.base:
            return   # base 是锚点，不更新
        st = self.state.setdefault(domain, {})
        n, s, ss = st.get(m, (0.0, 0.0, 0.0))
        g = self.gamma
        st[m] = (g * n + 1, g * s + reward, g * ss + reward * reward)


def build_capability_prior(cells, models, base, domain_of):
    """capability profile：每个 model 在"其它域"上的平均 reward(acc−base)，按 domain LODO。
    返回 {held_domain: {model: prior_mean}}——留出域的先验只用其它域，无泄漏。"""
    import collections
    by_dom_model = collections.defaultdict(lambda: collections.defaultdict(list))
    for (ds, N, seed), v in cells.items():
        if base not in v:
            continue
        for m in models:
            if m in v and m != base:
                by_dom_model[ds][m].append(v[m] - v[base])
    domains = sorted({k[0] for k in cells})
    # 每域 model 平均 reward
    dom_model_mean = {d: {m: float(np.mean(rs)) for m, rs in mm.items()} for d, mm in by_dom_model.items()}
    priors = {}
    for held in domains:
        agg = collections.defaultdict(list)
        for d, mm in dom_model_mean.items():
            if d == held:
                continue
            for m, mu in mm.items():
                agg[m].append(mu)
        priors[held] = {m: float(np.mean(v)) for m, v in agg.items()}
    return priors
