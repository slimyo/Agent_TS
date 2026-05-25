"""共享 LLM 客户端（复用 demo/.env 配置）。

读取 demo/.env 里的 PROVIDER + *_API_KEY，返回 OpenAI 兼容 client + 默认 model。
本地缓存按 (provider, model, prompt) hash 落盘到 research/.llm_cache/，
重复实验不重复调 API（同 seed 一致，节省费用）。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROVIDERS = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "env_key": "ZHIPU_API_KEY",
        # 默认用非 reasoning model（content 字段直接可用，速度更快）
        # 如需更强的 reasoning，可通过 .env 设置 MODEL=glm-4.7-flash 切换
        "model": "glm-4-flash-250414",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "env_key": "SILICONFLOW_API_KEY",
        # SiliconFlow 上的 GLM-4.6 名称
        "model": "zai-org/GLM-4.6",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "model": "qwen-plus",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env_key": "OLLAMA_API_KEY",
        "model": "qwen2.5:7b",
    },
}


_ROOT = Path(__file__).resolve().parents[2]
_CACHE = _ROOT / "research" / ".llm_cache"
_CACHE.mkdir(parents=True, exist_ok=True)


def make_client() -> tuple[OpenAI, str]:
    # demo/.env 优先；若不存在则用项目根目录的 .env
    for p in [_ROOT / "demo" / ".env", _ROOT / ".env"]:
        if p.exists():
            load_dotenv(p)
            break
    provider = os.getenv("PROVIDER", "zhipu").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown PROVIDER={provider}, choose from {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    key = os.getenv(cfg["env_key"])
    if not key:
        raise RuntimeError(f"set {cfg['env_key']} in .env (PROVIDER={provider})")
    client = OpenAI(api_key=key, base_url=cfg["base_url"])
    model = os.getenv("MODEL", cfg["model"])
    return client, model


def chat_cached(messages: list[dict], temperature: float = 0.1,
                max_tokens: int = 1024, **kwargs) -> str:
    """带磁盘缓存的 chat completion。同 prompt 第二次直接返回缓存。"""
    client, model = make_client()
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, **kwargs}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    fp = _CACHE / f"{key}.json"
    if fp.exists():
        return json.loads(fp.read_text())["content"]
    # 最多 5 次指数退避（zhipu / siliconflow 都有偶发 5xx）
    last_exc = None
    for attempt in range(5):
        try:
            # 给单次调用加 90s 硬超时，防止远端 hang 拖死整个 ablation
            resp = client.chat.completions.create(**payload, timeout=90)
            msg = resp.choices[0].message
            content = msg.content or ""
            if not content:
                content = getattr(msg, "reasoning_content", "") or ""
            fp.write_text(json.dumps({"payload": payload, "content": content}, ensure_ascii=False))
            return content
        except Exception as e:
            last_exc = e
            wait = 2 ** attempt + random.random()
            print(f"[llm] retry {attempt+1}/5 after error: {e!r} (sleep {wait:.1f}s)")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after 5 retries: {last_exc!r}")
