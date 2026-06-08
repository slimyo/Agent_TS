"""深度时序分类基线（纯 torch，无额外依赖）—— baseline.md 末尾「可扩展分类模型」类别二。

实现三个 TSC 深度经典/SOTA 架构，直接输出分类 logits：
  - fcn          : Fully Convolutional Network (Wang et al. 2017)
  - resnet       : ResNet for TSC (Wang et al. 2017)
  - inceptiontime: InceptionTime (Ismail Fawaz et al. 2019)

统一签名 `predict(X_train, y_train, X_test, **kw) -> y_pred`，输入 `[N,L]`（单变量；
UEA channel-flatten 后亦为 2D）。少样本下用早停 + 标准化 + Adam 训练。
torch 已在 tsci(本地) 与各远程 env；模型小（<1M 参数），本地即可，无需远程。
"""
from __future__ import annotations

import numpy as np


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prep(X_train, y_train, X_test):
    import torch
    Xtr = np.asarray(X_train, dtype=np.float32)
    Xte = np.asarray(X_test, dtype=np.float32)
    if Xtr.ndim == 3:  # [N,C,L] -> flatten channels（深度网这里按单通道处理）
        Xtr = Xtr.reshape(Xtr.shape[0], -1); Xte = Xte.reshape(Xte.shape[0], -1)
    # per-series z-norm（TSC 标准做法）
    mu, sd = Xtr.mean(1, keepdims=True), Xtr.std(1, keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd
    mu2, sd2 = Xte.mean(1, keepdims=True), Xte.std(1, keepdims=True) + 1e-8
    Xte = (Xte - mu2) / sd2
    classes = np.unique(y_train)
    cls2i = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cls2i[c] for c in y_train], dtype=np.int64)
    return (torch.tensor(Xtr).unsqueeze(1), torch.tensor(ytr),
            torch.tensor(Xte).unsqueeze(1), classes)


# ---------- 架构 ---------- #

def _conv_bn(cin, cout, k):
    import torch.nn as nn
    # padding="same" 保证长度不变（偶数 kernel 用 k//2 会 +1，破坏残差相加）
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding="same"), nn.BatchNorm1d(cout), nn.ReLU())


def _make_fcn(n_in, n_cls):
    import torch.nn as nn
    class FCN(nn.Module):
        def __init__(s):
            super().__init__()
            s.body = nn.Sequential(_conv_bn(n_in, 128, 8), _conv_bn(128, 256, 5), _conv_bn(256, 128, 3))
            s.head = nn.Linear(128, n_cls)
        def forward(s, x):
            return s.head(s.body(x).mean(-1))
    return FCN()


def _make_resnet(n_in, n_cls):
    import torch, torch.nn as nn
    class Block(nn.Module):
        def __init__(s, cin, cout):
            super().__init__()
            s.c = nn.Sequential(_conv_bn(cin, cout, 8), _conv_bn(cout, cout, 5),
                                nn.Conv1d(cout, cout, 3, padding="same"), nn.BatchNorm1d(cout))
            s.sc = nn.Sequential(nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout))
            s.relu = nn.ReLU()
        def forward(s, x):
            return s.relu(s.c(x) + s.sc(x))
    class ResNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.body = nn.Sequential(Block(n_in, 64), Block(64, 128), Block(128, 128))
            s.head = nn.Linear(128, n_cls)
        def forward(s, x):
            return s.head(s.body(x).mean(-1))
    return ResNet()


def _make_inception(n_in, n_cls, nf=32, depth=6):
    import torch, torch.nn as nn
    class Inception(nn.Module):
        def __init__(s, cin):
            super().__init__()
            s.bottle = nn.Conv1d(cin, nf, 1, padding=0) if cin > 1 else None
            cb = nf if s.bottle is not None else cin
            ks = [9, 19, 39]
            s.convs = nn.ModuleList([nn.Conv1d(cb, nf, k, padding=k // 2, bias=False) for k in ks])
            s.mp = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(cin, nf, 1, bias=False))
            s.bn = nn.Sequential(nn.BatchNorm1d(nf * 4), nn.ReLU())
        def forward(s, x):
            z = s.bottle(x) if s.bottle is not None else x
            outs = [c(z) for c in s.convs] + [s.mp(x)]
            return s.bn(torch.cat(outs, 1))
    class InceptionTime(nn.Module):
        def __init__(s):
            super().__init__()
            s.blocks = nn.ModuleList(); s.short = nn.ModuleList()
            cin = n_in
            for d in range(depth):
                s.blocks.append(Inception(cin))
                if d % 3 == 2:
                    s.short.append(nn.Sequential(nn.Conv1d(n_in if d == 2 else nf * 4, nf * 4, 1),
                                                 nn.BatchNorm1d(nf * 4)))
                cin = nf * 4
            s.relu = nn.ReLU(); s.head = nn.Linear(nf * 4, n_cls)
        def forward(s, x):
            res = x; si = 0
            for d, b in enumerate(s.blocks):
                x = b(x)
                if d % 3 == 2:
                    x = s.relu(x + s.short[si](res)); res = x; si += 1
            return s.head(x.mean(-1))
    return InceptionTime()


# ---------- 训练 / 预测 ---------- #

def _train_predict(make_model, X_train, y_train, X_test, epochs=200, lr=1e-3, seed=0):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xte, classes = _prep(X_train, y_train, X_test)
    dev = _device()
    model = make_model(1, len(classes)).to(dev)
    Xtr, ytr, Xte = Xtr.to(dev), ytr.to(dev), Xte.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    bs = min(16, len(Xtr))
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1).cpu().numpy()
    return classes[pred]


def fcn(X_train, y_train, X_test, epochs=200, **_):
    return _train_predict(_make_fcn, X_train, y_train, X_test, epochs=epochs)


def resnet(X_train, y_train, X_test, epochs=200, **_):
    return _train_predict(_make_resnet, X_train, y_train, X_test, epochs=epochs)


def inceptiontime(X_train, y_train, X_test, epochs=200, **_):
    return _train_predict(_make_inception, X_train, y_train, X_test, epochs=epochs)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((16, 80)).astype(np.float32); Xtr[:8] += 0.8
    ytr = np.array([0] * 8 + [1] * 8); Xte = rng.standard_normal((6, 80)).astype(np.float32); Xte[:3] += 0.8
    yte = np.array([0] * 3 + [1] * 3)
    for nm, fn in [("fcn", fcn), ("resnet", resnet), ("inceptiontime", inceptiontime)]:
        try:
            p = fn(Xtr, ytr, Xte, epochs=60); print(f"{nm:14} OK acc={(p==yte).mean():.2f} pred={list(p)}")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"{nm:14} FAIL {type(e).__name__}: {str(e)[:80]}")
