"""task #17 / B5 · Dataset-name 语义先验 for Curator/Agent。

提供每个 dataset 的 domain 描述、典型 pattern、known failure modes，让 LLM 在
做诊断 / RCA 时有 task-level 世界知识。

Source：domain knowledge + 论文/数据集 README。
"""
from __future__ import annotations

DATASET_PRIORS: dict[str, dict] = {
    "ETTh1": {
        "domain": "electricity transformer station, hourly sampling",
        "typical_patterns": "strong daily seasonality (24h), weak weekly (168h); "
                           "temperature drives 'oil temperature' (OT) target column",
        "known_quirks": "regional temperature regime shifts on long horizons; "
                        "weekend/weekday demand differences",
        "season_m": 24,
        "scale_range": "0 to 80 (deg C of oil temperature)",
    },
    "ETTh2": {
        "domain": "electricity transformer station 2, hourly sampling",
        "typical_patterns": "similar to ETTh1 but different transformer + load profile",
        "known_quirks": "more volatile than ETTh1 due to industrial load",
        "season_m": 24,
        "scale_range": "0 to 60",
    },
    "ECL": {
        "domain": "electricity consumption (321 customers MT_001 first column)",
        "typical_patterns": "daily 24h + weekly 168h patterns; high baseline + spikes",
        "known_quirks": "customer behavior heterogeneous; holidays cause discontinuity",
        "season_m": 24,
        "scale_range": "0 to ~100 (kWh)",
    },
    "Exchange": {
        "domain": "foreign exchange rates, daily",
        "typical_patterns": "near-martingale / random walk; little periodicity",
        "known_quirks": "regime shifts on macro events; no strong trend in short windows",
        "season_m": 7,  # weekly trading
        "scale_range": "0.5 to 1.7 (currency ratios)",
    },
    "Weather": {
        "domain": "meteorology multivariate, 10-min sampling (OT = wind speed)",
        "typical_patterns": "strong daily 144-step + seasonal patterns; "
                           "diurnal temperature, weather-system propagation",
        "known_quirks": "rapid local shifts on storms; multi-modal distribution",
        "season_m": 144,
        "scale_range": "0 to ~100 (wind / temp depending on column)",
    },
    "ILI": {
        "domain": "national influenza-like illness weekly counts (OT column)",
        "typical_patterns": "strong annual seasonality (m=52), winter peak",
        "known_quirks": "pandemic years break baseline; reporting lags",
        "season_m": 52,
        "scale_range": "0 to ~5000 (case count, can spike to 10000+)",
    },
    "Coffee": {
        "domain": "coffee variety spectroscopy (UCR-classic)",
        "typical_patterns": "smooth absorbance curves; class diff in mid-range peaks",
        "known_quirks": "very small N (28 train), highly discriminative spectra",
        "season_m": 1,
    },
    "ECG200": {
        "domain": "ECG normal vs ischemia (binary)",
        "typical_patterns": "QRS complex shape, T-wave morphology",
        "known_quirks": "short window (96 length); class boundary at morphology",
        "season_m": 1,
    },
    "BeetleFly": {
        "domain": "insect silhouette outlines",
        "typical_patterns": "smooth closed contours; class = wing vs body shape",
        "known_quirks": "image-derived, no real time semantic",
        "season_m": 1,
    },
    "BirdChicken": {
        "domain": "bird vs chicken silhouette outlines",
        "typical_patterns": "global shape morphology",
        "known_quirks": "20 train, image-outline (visual)",
        "season_m": 1,
    },
}


def get_prior(name: str) -> dict | None:
    return DATASET_PRIORS.get(name)


def render_prior_block(name: str) -> str:
    """生成可插入 LLM prompt 的 prior 文本。"""
    p = DATASET_PRIORS.get(name)
    if p is None:
        return ""
    return (f"【Dataset semantic prior】 {name}\n"
            f"- Domain: {p['domain']}\n"
            f"- Typical patterns: {p['typical_patterns']}\n"
            f"- Known quirks: {p.get('known_quirks', '(n/a)')}\n"
            f"- Season period m: {p['season_m']}\n"
            + (f"- Scale: {p['scale_range']}\n" if 'scale_range' in p else "")
            + "**Use this prior to judge what is 'normal' vs 'anomalous' on this dataset.**\n")


if __name__ == "__main__":
    for name in ["ETTh1", "Weather", "Coffee"]:
        print(render_prior_block(name))
        print("---")
