"""Round 6 B3 · Drift Engine (feedback 后§6 "你现在没有真正的 Drift Engine").

Computes 4 drift signals from RouterState and recommends remediation actions:

    Signal               Source                       Action when triggered
    ─────────────────────────────────────────────────────────────────────────
    feature_drift_kl     memory_cases.z (projected)   boost_exploration
    residual_drift_ks    telemetry.outcome            lower_memory_trust
    routing_drift_kl     telemetry.chosen             boost_exploration
    memory_mismatch      outcome vs per-regime mean   lower_memory_trust
                                                     + mark_regime_stale

Flow:
    signals = compute_drift(state, config)
    actions = recommend_actions(signals, config)
    applied = apply_actions(state, actions)

Pure-functional `compute_drift` + `recommend_actions`; side-effects isolated
in `apply_actions` (mutates state.memory_trust, state.regime_stale,
state.bandit.decay_scale via setattr — does not touch persistent fields, so
existing save/load remain compatible).

Consumers (MemoryLikelihood, BanditLikelihoodFactor) read these via
`getattr(state, "memory_trust", 1.0)` etc. — no hard dependency.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import Counter
import math

import numpy as np


# ─── Config & data classes ───────────────────────────────────────────────────


@dataclass
class DriftConfig:
    window_recent: int = 30         # records considered "recent"
    window_history: int = 90        # records considered "historical baseline"
    th_feature_kl: float = 0.5
    th_residual_ks: float = 0.30
    th_routing_kl: float = 0.5
    th_memory_mismatch: float = 0.40
    th_pred_residual_z: float = 2.0   # F-R6.1 fix: routing-independent mean shift
    smoothing: float = 0.5          # Laplace smoothing for categorical KL
    z_proj_dim: int = 0             # which coordinate of z to project for feature KL
    n_bins: int = 8                 # histogram bins for continuous KL


@dataclass
class DriftSignals:
    feature_kl: float
    residual_ks: float
    routing_kl: float
    memory_mismatch: float
    pred_residual_z: float          # F-R6.1: |E[recent] − E[hist]| / σ_hist
    detected: dict[str, bool] = field(default_factory=dict)
    window_recent_used: int = 0
    window_history_used: int = 0

    def any_detected(self) -> bool:
        return any(self.detected.values())


@dataclass
class DriftAction:
    kind: str           # "boost_exploration" | "lower_memory_trust" | "mark_regime_stale"
    magnitude: float    # ∈ (0, 1] strength; larger = stronger response
    reasons: list[str] = field(default_factory=list)


# ─── Signal computations ─────────────────────────────────────────────────────


def _kl_categorical(p: dict, q: dict, smoothing: float = 0.5) -> float:
    keys = set(p) | set(q)
    if not keys: return 0.0
    sp = sum(p.values()) + smoothing * len(keys)
    sq = sum(q.values()) + smoothing * len(keys)
    out = 0.0
    for k in keys:
        pi = (p.get(k, 0) + smoothing) / sp
        qi = (q.get(k, 0) + smoothing) / sq
        out += pi * math.log(pi / qi)
    return out


def _kl_histogram(recent: np.ndarray, historical: np.ndarray,
                   n_bins: int = 8, smoothing: float = 0.5) -> float:
    if recent.size == 0 or historical.size == 0: return 0.0
    lo = float(min(recent.min(), historical.min()))
    hi = float(max(recent.max(), historical.max()))
    if hi - lo < 1e-9: return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    hr, _ = np.histogram(recent, bins=edges)
    hh, _ = np.histogram(historical, bins=edges)
    p = dict(enumerate(hr.tolist()))
    q = dict(enumerate(hh.tolist()))
    return _kl_categorical(p, q, smoothing)


def _ks_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    """Empirical Kolmogorov-Smirnov statistic ∈ [0, 1]."""
    if a.size == 0 or b.size == 0: return 0.0
    all_vals = np.concatenate([a, b])
    grid = np.sort(np.unique(all_vals))
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _extract_z_proj(cases: list[dict], dim: int = 0) -> np.ndarray:
    out = []
    for c in cases:
        z = c.get("z")
        if z is None: continue
        try:
            v = float(z[dim]) if hasattr(z, "__getitem__") and len(z) > dim else None
        except Exception:
            v = None
        if v is not None and math.isfinite(v):
            out.append(v)
    return np.asarray(out, dtype=np.float64)


def compute_drift(state, config: Optional[DriftConfig] = None) -> DriftSignals:
    """Compute all 4 drift signals from state. Pure function (no mutation)."""
    if config is None: config = DriftConfig()

    tel = list(state.telemetry)
    cases = list(state.memory_cases)

    # ─── 1. Feature drift (z[dim] histogram KL) ────────────────────────────
    z_all = _extract_z_proj(cases, dim=config.z_proj_dim)
    n_r = min(len(z_all), config.window_recent)
    n_h = min(max(len(z_all) - n_r, 0), config.window_history)
    if n_r >= 5 and n_h >= 5:
        z_recent = z_all[-n_r:]
        z_hist   = z_all[-(n_r + n_h):-n_r]
        feature_kl = _kl_histogram(z_recent, z_hist,
                                    n_bins=config.n_bins,
                                    smoothing=config.smoothing)
    else:
        feature_kl = 0.0

    # ─── 2. Residual drift (outcome KS) + 5. Pred-residual mean shift ──────
    # Both signals are *routing-independent* — they only need historical
    # outcomes, so they fire even when the router collapses to one model
    # (F-R6.1 fix from method2.md §11.5.5).
    outcomes = np.array([r.outcome for r in tel if r.outcome is not None],
                        dtype=np.float64)
    n_or = min(outcomes.size, config.window_recent)
    n_oh = min(max(outcomes.size - n_or, 0), config.window_history)
    if n_or >= 5 and n_oh >= 5:
        rec_o  = outcomes[-n_or:]
        hist_o = outcomes[-(n_or + n_oh):-n_or]
        residual_ks = _ks_two_sample(rec_o, hist_o)
        sigma_h = float(np.std(hist_o) + 1e-6)
        pred_residual_z = float(abs(np.mean(rec_o) - np.mean(hist_o)) / sigma_h)
    else:
        residual_ks = 0.0
        pred_residual_z = 0.0

    # ─── 3. Routing drift (chosen-model dist KL) ───────────────────────────
    n_tr = min(len(tel), config.window_recent)
    n_th = min(max(len(tel) - n_tr, 0), config.window_history)
    if n_tr >= 5 and n_th >= 5:
        rec_choices  = Counter(r.chosen for r in tel[-n_tr:])
        hist_choices = Counter(r.chosen for r in tel[-(n_tr + n_th):-n_tr])
        routing_kl = _kl_categorical(dict(rec_choices), dict(hist_choices),
                                      config.smoothing)
    else:
        routing_kl = 0.0

    # ─── 4. Memory mismatch (per-regime outcome shock rate) ────────────────
    # Build per-regime hist mean/std from history half; count recent records whose
    # outcome falls outside ±2σ.
    by_regime_hist: dict = {}
    if n_th >= 5:
        for r in tel[-(n_tr + n_th):-n_tr]:
            if r.outcome is None: continue
            rg = r.ctx_summary.get("regime")
            by_regime_hist.setdefault(rg, []).append(r.outcome)
    stats = {rg: (float(np.mean(v)), float(np.std(v) + 1e-6))
             for rg, v in by_regime_hist.items() if len(v) >= 3}
    if stats and n_tr >= 5:
        misses, total = 0, 0
        for r in tel[-n_tr:]:
            if r.outcome is None: continue
            rg = r.ctx_summary.get("regime")
            if rg not in stats: continue
            mu, sd = stats[rg]
            total += 1
            if abs(r.outcome - mu) > 2 * sd: misses += 1
        memory_mismatch = float(misses / total) if total else 0.0
    else:
        memory_mismatch = 0.0

    detected = {
        "feature":       feature_kl       > config.th_feature_kl,
        "residual":      residual_ks      > config.th_residual_ks,
        "routing":       routing_kl       > config.th_routing_kl,
        "memory":        memory_mismatch  > config.th_memory_mismatch,
        "pred_residual": pred_residual_z  > config.th_pred_residual_z,
    }

    return DriftSignals(
        feature_kl=round(feature_kl, 4),
        residual_ks=round(residual_ks, 4),
        routing_kl=round(routing_kl, 4),
        memory_mismatch=round(memory_mismatch, 4),
        pred_residual_z=round(pred_residual_z, 4),
        detected=detected,
        window_recent_used=n_tr,
        window_history_used=n_th,
    )


# ─── Action recommendation ───────────────────────────────────────────────────


def recommend_actions(signals: DriftSignals,
                      config: Optional[DriftConfig] = None) -> list[DriftAction]:
    """Map detected signals to remediation actions.

    Mapping rationale:
      feature_kl  → input domain shift   ⇒ explore harder
      residual_ks → output regime shift  ⇒ memory likely stale ⇒ lower trust
      routing_kl  → choice mix shifting  ⇒ explore harder
      memory_mismatch → per-regime model rankings invalid
                       ⇒ lower trust + mark regime stale (retrain)
    """
    if config is None: config = DriftConfig()
    actions: list[DriftAction] = []

    # boost_exploration ← feature + routing + pred_residual (F-R6.1)
    explore_reasons, explore_mag = [], 0.0
    if signals.detected.get("feature"):
        explore_reasons.append(f"feature_kl={signals.feature_kl:.3f}")
        explore_mag = max(explore_mag, signals.feature_kl / max(config.th_feature_kl, 1e-6))
    if signals.detected.get("routing"):
        explore_reasons.append(f"routing_kl={signals.routing_kl:.3f}")
        explore_mag = max(explore_mag, signals.routing_kl / max(config.th_routing_kl, 1e-6))
    if signals.detected.get("pred_residual"):
        explore_reasons.append(f"pred_residual_z={signals.pred_residual_z:.2f}")
        explore_mag = max(explore_mag,
                          signals.pred_residual_z / max(config.th_pred_residual_z, 1e-6))
    if explore_reasons:
        actions.append(DriftAction(
            kind="boost_exploration",
            magnitude=min(1.0, explore_mag),
            reasons=explore_reasons,
        ))

    # lower_memory_trust ← residual + memory_mismatch + pred_residual (F-R6.1)
    trust_reasons, trust_mag = [], 0.0
    if signals.detected.get("residual"):
        trust_reasons.append(f"residual_ks={signals.residual_ks:.3f}")
        trust_mag = max(trust_mag, signals.residual_ks / max(config.th_residual_ks, 1e-6))
    if signals.detected.get("memory"):
        trust_reasons.append(f"memory_mismatch={signals.memory_mismatch:.3f}")
        trust_mag = max(trust_mag, signals.memory_mismatch / max(config.th_memory_mismatch, 1e-6))
    if signals.detected.get("pred_residual"):
        trust_reasons.append(f"pred_residual_z={signals.pred_residual_z:.2f}")
        trust_mag = max(trust_mag,
                        signals.pred_residual_z / max(config.th_pred_residual_z, 1e-6))
    if trust_reasons:
        actions.append(DriftAction(
            kind="lower_memory_trust",
            magnitude=min(1.0, trust_mag),
            reasons=trust_reasons,
        ))

    # mark_regime_stale ← memory_mismatch only (rankings inside a regime invalid)
    if signals.detected.get("memory"):
        actions.append(DriftAction(
            kind="mark_regime_stale",
            magnitude=min(1.0, signals.memory_mismatch /
                          max(config.th_memory_mismatch, 1e-6)),
            reasons=[f"memory_mismatch={signals.memory_mismatch:.3f}"],
        ))

    return actions


# ─── Action application ──────────────────────────────────────────────────────


def refit_regimes(state, K: Optional[int] = None,
                  recent_n: int = 300,
                  min_cases: int = 30,
                  bandit_discount: float = 0.3) -> dict:
    """Refit regime centroids + per-regime priors from recent memory_cases.

    Called when `state.regime_stale` is True (mark_regime_stale fired). Side
    effects on `state`:
        - state.regime_centroids ← refitted, L2-normalized
        - state.regime_priors    ← recomputed from per-(regime, chosen) losses
        - state.bandit._state    ← all (n, s, sq) scaled by `bandit_discount`
                                   so prior μ persists but new signal dominates
        - state.regime_stale     ← False (cleared)

    Returns a summary dict with status.
    """
    summary: dict = {"status": "skipped", "reason": ""}
    cases = list(state.memory_cases)[-recent_n:]
    Z, winners = [], []
    for c in cases:
        z = c.get("z")
        chosen = c.get("chosen")
        outcome = c.get("outcome")
        if z is None or chosen is None or outcome is None: continue
        try:
            zv = np.asarray(z, dtype=np.float64)
            if not np.all(np.isfinite(zv)): continue
        except Exception:
            continue
        Z.append(zv); winners.append({chosen: float(outcome)})

    if len(Z) < min_cases:
        summary["reason"] = f"insufficient cases ({len(Z)} < {min_cases})"
        return summary

    Z_arr = np.stack(Z, axis=0)
    # Determine K: match current K when available, else default 8
    if K is None:
        K = (int(state.regime_centroids.shape[0])
             if state.regime_centroids is not None else 8)
    K = max(2, min(K, len(Z_arr)))

    from research.agent.representation import RegimeAssigner
    assigner = RegimeAssigner(K=K)
    assigner.fit(Z_arr, winners)
    if assigner._centroids is None:
        summary["reason"] = "KMeans returned no centroids"
        return summary

    # L2-normalize centroids so cosine dot-product matches embedding convention
    cents = assigner._centroids.astype(np.float64)
    norms = np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12
    state.regime_centroids = cents / norms
    state.regime_priors = {int(r): dict(pi)
                            for r, pi in assigner._per_regime_winners.items()}

    # Soft-reset bandit: regime ids carry different semantics now.
    if hasattr(state, "bandit") and hasattr(state.bandit, "_state"):
        d = float(max(0.0, min(1.0, bandit_discount)))
        state.bandit._state = {k: (n * d, s * d, sq * d)
                                for k, (n, s, sq) in state.bandit._state.items()}

    state.regime_stale = False
    summary.update({
        "status": "ok", "K": K, "n_cases_used": len(Z_arr),
        "bandit_discount": bandit_discount,
        "regime_priors_K": len(state.regime_priors),
    })
    return summary


def apply_actions(state, actions: list[DriftAction]) -> dict:
    """Mutate `state` to reflect drift remediation.

    Stored as plain attributes (no dataclass fields), so save/load remain
    compatible. Consumers should read with `getattr(state, name, default)`.

    Attributes set:
        state.memory_trust ∈ (0, 1]      — multiplier on MemoryLikelihood weight
        state.regime_stale: bool         — regime centroids need retrain
        state.bandit_explore_scale ≥ 1   — multiplier on bandit exploration noise

    Returns a dict summary for telemetry/logging.
    """
    summary = {"applied": [], "memory_trust": getattr(state, "memory_trust", 1.0),
               "regime_stale": getattr(state, "regime_stale", False),
               "bandit_explore_scale": getattr(state, "bandit_explore_scale", 1.0)}
    for a in actions:
        if a.kind == "boost_exploration":
            new_scale = 1.0 + 2.0 * a.magnitude     # up to 3× exploration
            state.bandit_explore_scale = max(
                getattr(state, "bandit_explore_scale", 1.0), new_scale)
            summary["bandit_explore_scale"] = state.bandit_explore_scale
            # Round 8 M4 · also tighten per-regime decay for recently active
            # regimes so they forget faster post-drift.
            if hasattr(state, "bandit") and hasattr(state.bandit,
                                                      "set_regime_decay"):
                recent = set()
                for rec in list(state.telemetry)[-30:]:
                    rg = rec.ctx_summary.get("regime")
                    if rg is not None: recent.add(rg)
                tighten_to = max(0.85, 1.0 - 0.10 * a.magnitude)
                for rg in recent:
                    cur = state.bandit._effective_decay(rg)
                    state.bandit.set_regime_decay(rg, min(cur, tighten_to))
                if recent:
                    summary["regime_decay_tightened"] = {
                        int(rg): round(state.bandit._effective_decay(rg), 4)
                        for rg in recent}
        elif a.kind == "lower_memory_trust":
            new_trust = 1.0 - 0.7 * a.magnitude     # floor 0.3
            state.memory_trust = min(
                getattr(state, "memory_trust", 1.0), new_trust)
            summary["memory_trust"] = state.memory_trust
        elif a.kind == "mark_regime_stale":
            state.regime_stale = True
            summary["regime_stale"] = True
        summary["applied"].append({"kind": a.kind, "mag": a.magnitude,
                                    "reasons": a.reasons})
    return summary


# ─── one-shot helper ─────────────────────────────────────────────────────────


def run_drift_step(state, config: Optional[DriftConfig] = None,
                   apply: bool = True,
                   auto_refit: bool = True,
                   log_event: bool = True) -> dict:
    """Compute → recommend → (optionally) apply → (optionally) refit → log."""
    import time as _time
    signals = compute_drift(state, config)
    actions = recommend_actions(signals, config)
    out = {"signals": asdict(signals), "actions": [asdict(a) for a in actions]}
    if apply:
        out["applied"] = apply_actions(state, actions)
        if auto_refit and getattr(state, "regime_stale", False):
            out["refit"] = refit_regimes(state)

    # Round 6 B3 · persistent event log (only when something happened OR
    # apply requested — keeps log informative without flooding).
    if log_event and (signals.any_detected() or actions):
        event = {
            "t": _time.time(),
            "n_observations": int(state.n_observations),
            "n_decisions":    int(state.n_decisions),
            "signals":        out["signals"],
            "actions":        [{"kind": a["kind"], "mag": a["magnitude"],
                                 "reasons": a["reasons"]} for a in out["actions"]],
            "applied":        out.get("applied", {}).get("applied", []) if apply else [],
            "memory_trust":   float(getattr(state, "memory_trust", 1.0)),
            "explore_scale":  float(getattr(state, "bandit_explore_scale", 1.0)),
            "regime_stale":   bool(getattr(state, "regime_stale", False)),
            "refit_status":   out.get("refit", {}).get("status"),
        }
        if not hasattr(state, "drift_history"):
            state.drift_history = []
        state.drift_history.append(event)
        out["event_logged"] = True
    return out


# ─── smoke ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time, random
    from research.agent.router_state import RouterState, TelemetryRecord

    print("=" * 60)
    print("Round 6 B3 · Drift Engine smoke")
    print("=" * 60)

    state = RouterState()
    random.seed(0); np.random.seed(0)
    models = ["chronos2", "tirex", "toto", "naive_drift"]

    # Stable phase: 200 records under regime 0/1, chronos2 dominates, low outcomes
    for i in range(200):
        regime = i % 2
        chosen = "chronos2" if random.random() < 0.7 else random.choice(models)
        post = {m: 0.1 for m in models}; post[chosen] = 0.7
        s = sum(post.values()); post = {m: v/s for m, v in post.items()}
        rec = TelemetryRecord(
            t=time.time() - (400 - i) * 60,
            ctx_summary={"dataset": "ETTh1", "N": 50, "H": 24, "regime": regime},
            chosen=chosen, posterior=post,
            prior_contribs={}, lik_contribs={},
            decide_mode="argmax",
            outcome=float(np.random.normal(0.30, 0.05)),
        )
        state.telemetry.append(rec)
        state.memory_cases.append({
            "t": rec.t, "chosen": chosen, "outcome": rec.outcome, "regime": regime,
            "z": np.random.normal(0.0, 1.0, size=3).tolist(),
        })

    # Drifted phase: 60 recent records — regime hops to 2, outcomes worse,
    # chosen distribution flips to tirex, z mean shifts.
    for i in range(60):
        regime = 2
        chosen = "tirex" if random.random() < 0.7 else "toto"
        post = {m: 0.1 for m in models}; post[chosen] = 0.7
        s = sum(post.values()); post = {m: v/s for m, v in post.items()}
        rec = TelemetryRecord(
            t=time.time() - (60 - i) * 60,
            ctx_summary={"dataset": "ETTh1", "N": 50, "H": 24, "regime": regime},
            chosen=chosen, posterior=post,
            prior_contribs={}, lik_contribs={},
            decide_mode="argmax",
            outcome=float(np.random.normal(0.80, 0.20)),   # ← shock
        )
        state.telemetry.append(rec)
        state.memory_cases.append({
            "t": rec.t, "chosen": chosen, "outcome": rec.outcome, "regime": regime,
            "z": np.random.normal(3.0, 1.0, size=3).tolist(),   # ← shifted z
        })
    state.n_decisions = len(state.telemetry)
    state.n_observations = sum(1 for r in state.telemetry if r.outcome is not None)

    cfg = DriftConfig()
    sig = compute_drift(state, cfg)
    print(f"\n[Signals]")
    print(f"  feature_kl       = {sig.feature_kl:.3f}  detected={sig.detected['feature']}")
    print(f"  residual_ks      = {sig.residual_ks:.3f}  detected={sig.detected['residual']}")
    print(f"  routing_kl       = {sig.routing_kl:.3f}  detected={sig.detected['routing']}")
    print(f"  memory_mismatch  = {sig.memory_mismatch:.3f}  detected={sig.detected['memory']}")
    print(f"  windows: recent={sig.window_recent_used}  hist={sig.window_history_used}")

    actions = recommend_actions(sig, cfg)
    print(f"\n[Actions] ({len(actions)})")
    for a in actions:
        print(f"  - {a.kind}  mag={a.magnitude:.2f}  reasons={a.reasons}")

    summary = apply_actions(state, actions)
    print(f"\n[Applied]")
    print(f"  memory_trust         = {summary['memory_trust']:.3f}")
    print(f"  bandit_explore_scale = {summary['bandit_explore_scale']:.3f}")
    print(f"  regime_stale         = {summary['regime_stale']}")
