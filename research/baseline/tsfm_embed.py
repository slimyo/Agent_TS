"""TSFM 嵌入 + 线性探头分类 —— baseline.md 末尾「可扩展分类模型」类别三。

把预训练时序基础模型的编码器隐状态当作分类特征，接一个 LogisticRegression 线性探头
（与现有 moment_1nn/mantis_lr 平行，但用更大的生成式 TSFM）。

统一签名 `predict(X_train, y_train, X_test, **kw) -> y_pred`，输入 `[N,L]`（多变量先 flatten）。
**大权重 → 在各自远程 env 跑**（见 baseline.md §4.2）：
  - chronos2_emb : tsci-c2        (amazon/chronos-2 encoder embed)
  - timesfm_emb  : tsci-tsfm      (google/timesfm-2.0 stacked-transformer 输出)
  - timer_emb    : tsci-remote    (Timer-S1 decoder hidden states, TIMER_FORCE_GPU=1)
"""
from __future__ import annotations

import numpy as np

_CAP = 512   # 上下文截断，控显存


def _retry(fn, tries=5, wait=3.0):
    """模型加载重试：HF 偶发连接抖动（即便 offline）→ 缓存重试通常立即成功。
    全部失败才抛（让 _safe 退 majority），避免单次抖动污染整格。"""
    import time
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(wait)
    raise last


def _lr_probe(emb_fn, X_train, y_train, X_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Etr = np.asarray(emb_fn(X_train), dtype=np.float64)
    Ete = np.asarray(emb_fn(X_test), dtype=np.float64)
    sc = StandardScaler().fit(Etr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Etr), np.asarray(y_train))
    return np.asarray(clf.predict(sc.transform(Ete)))


def _flat(X):
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(X.shape[0], -1) if X.ndim == 3 else X


def _cap(x):
    return x[-_CAP:] if len(x) > _CAP else x


# ---------- chronos-2 编码器嵌入 ---------- #

def _chronos2_embed(X):
    import torch
    from research.baseline.chronos2 import _get_pipeline
    pipe = _retry(_get_pipeline)
    out = []
    for x in _flat(X):
        ctx = torch.tensor(_cap(x), dtype=torch.float32)
        emb, _ = pipe.embed(ctx)          # [1, seq, D]
        out.append(emb[0].float().mean(0).cpu().numpy())
    return np.stack(out)


def chronos2_emb(X_train, y_train, X_test, **_):
    return _lr_probe(_chronos2_embed, X_train, y_train, X_test)


# ---------- Timer-S1 解码器隐状态 ---------- #

def _timer_embed(X):
    import torch
    from research.baseline.timer import _get
    model = _retry(_get)
    out = []
    for x in _flat(X):
        ctx = torch.tensor(_cap(x), dtype=model.dtype, device=model.device).reshape(1, -1)
        with torch.no_grad():
            res = model(ctx, output_hidden_states=True)
        h = res.hidden_states[-1] if hasattr(res, "hidden_states") and res.hidden_states is not None \
            else (res[0] if isinstance(res, (tuple, list)) else res.last_hidden_state)
        out.append(h[0].float().mean(0).cpu().numpy())
    return np.stack(out)


def timer_emb(X_train, y_train, X_test, **_):
    return _lr_probe(_timer_embed, X_train, y_train, X_test)


# ---------- TimesFM-2.0 stacked-transformer 嵌入 ---------- #

def _timesfm_embed(X, H=16):
    """真·TimesFM 推理派生特征：一次 batched forecast(所有序列) → 每序列用
    [point_forecast(H 维) + 输入 mean/std/last/min/max] 作特征。
    用 timesfm.TimesFm.forecast(inputs=[...]) 的原生批处理（一 cell 一次调用，快）。"""
    from research.baseline import timesfm2 as T
    tfm = _retry(lambda: T._get_pipeline(H))   # 真正加载 TimesFM 模型（带重试）
    series = [_cap(x).astype(np.float32) for x in _flat(X)]
    point, _ = tfm.forecast(inputs=series, freq=[0] * len(series))  # [n, H] 真实预测
    point = np.asarray(point)[:, :H]
    out = []
    for i, x in enumerate(series):
        stats = [float(x.mean()), float(x.std()), float(x[-1]), float(x.min()), float(x.max())]
        out.append(np.concatenate([point[i], stats]))
    return np.stack(out)


def timesfm_emb(X_train, y_train, X_test, **_):
    return _lr_probe(_timesfm_embed, X_train, y_train, X_test)
