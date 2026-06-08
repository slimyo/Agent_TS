"""远程 TSFM 可用性盘点：逐个 forecasting baseline 在小序列上 load+predict，报 OK/缺权重/缺依赖/报错。

用法（远程，两套 env 各跑一次）：
  HF_HOME=... HF_ENDPOINT=https://hf-mirror.com TIMER_FORCE_GPU=1 python -m research.experiments.tsfm_inventory_smoke
输出：stdout 表格 + research/results/tsfm_inventory.jsonl（含 env 标记）
"""
from __future__ import annotations
import importlib, json, os, time, traceback
from pathlib import Path
import numpy as np

MODELS = ["chronos", "chronos2", "chronos_bolt", "moirai2", "sundial",
          "time_moe", "timer", "timesfm2", "tirex", "toto", "toto2"]
H = 24


def classify_err(e: BaseException) -> str:
    s = f"{type(e).__name__}: {e}".lower()
    if any(k in s for k in ["no module named", "modulenotfound", "cannot import", "importerror"]):
        return "MISSING_DEP"
    if any(k in s for k in ["not found", "does not appear", "404", "couldn't connect",
                            "connectionerror", "offline", "no such file", "repository not found"]):
        return "MISSING_WEIGHTS"
    if "out of memory" in s or "cuda error" in s:
        return "OOM"
    return "ERROR"


def main():
    env_tag = os.environ.get("ENV_TAG", "?")
    rng = np.random.default_rng(0)
    sig = (np.sin(np.arange(400) * 0.1) + 0.1 * rng.standard_normal(400)).astype(np.float32)
    val = np.array([], dtype=np.float32)
    out = Path("research/results/tsfm_inventory.jsonl")
    rows = []
    print(f"=== TSFM inventory (env={env_tag}) ===", flush=True)
    for m in MODELS:
        rec = {"env": env_tag, "model": m}
        try:
            mod = importlib.import_module(f"research.baseline.{m}")
            t0 = time.time()
            p = mod.predict(sig, val, H=H, seed=1)
            dt = time.time() - t0
            ok = hasattr(p, "__len__") and len(p) >= 1 and np.isfinite(np.asarray(p, float)).any()
            rec.update(status="OK" if ok else "BAD_OUTPUT",
                       pred_len=int(len(p)) if hasattr(p, "__len__") else 0,
                       wall=round(dt, 1))
        except BaseException as e:  # noqa: BLE001
            rec.update(status=classify_err(e), err=f"{type(e).__name__}: {str(e)[:160]}")
        rows.append(rec)
        print(f"  {m:13} {rec['status']:15} {rec.get('err','')}"[:160], flush=True)

    # 追加写（两个 env 各一遍）
    with out.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    oks = [r["model"] for r in rows if r["status"] == "OK"]
    print(f"\nenv={env_tag} OK ({len(oks)}/{len(MODELS)}): {oks}", flush=True)


if __name__ == "__main__":
    main()
