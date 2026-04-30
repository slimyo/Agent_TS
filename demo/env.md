# 环境与代理配置笔记

记录本机（WSL2 Ubuntu）跑本 demo 的环境配置，特别是**保留 Windows 代理 + 国内 LLM 域名直连**的设置。

---

## 1. Python 环境

- 工具：**mamba** 2.5.0（conda 26.3.2 装在 `/root/Downloads/ENTER/`）
- 项目环境：`tsci`（Python 3.10）
- 创建命令：

  ```bash
  mamba create -n tsci python=3.10 -y
  mamba activate tsci
  cd /home/hz/code/agent_ts/demo
  pip install -r requirements.txt
  ```

- 每次开终端记得 `mamba activate tsci`
- VSCode：`Ctrl+Shift+P` → `Python: Select Interpreter` → 选 `envs/tsci/bin/python`

---

## 2. 系统代理现状

WSL 镜像了 Windows 的 Clash/v2ray 代理：

```
http_proxy = http://127.0.0.1:7890
https_proxy = http://127.0.0.1:7890
all_proxy = socks5://127.0.0.1:7890
```

需要走代理的场景（github / pypi / anthropic / openai 等海外服务）必须保留这套设置。

## 3. 遇到的问题

跑 `01_curator_minimal.py` 时报错：

```
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
Make sure to install httpx using `pip install httpx[socks]`.
```

**原因**：`openai` SDK 走 httpx；httpx 看到 `all_proxy=socks5://...` 后试图初始化 SOCKS transport，但缺 `socksio` 包。

**两种修法**：

| 方案 | 命令 | 说明 |
|---|---|---|
| **装 socks 支持（已采用）** | `pip install "httpx[socks]"` | 让 httpx 能初始化 SOCKS transport |
| 加 `NO_PROXY` 白名单（已采用） | 见 §4 | 让国内域名跳过代理直连 |

**两步都要做**，原因见下。

### 3.1 关键坑：NO_PROXY 不能单独解决这个错

最初以为加 `NO_PROXY=open.bigmodel.cn` 就够了。实测**仍然报同样的错**。

原因：httpx 在 **OpenAI 客户端构造时**就要根据 `all_proxy` 初始化 SOCKS transport（这一步就缺 `socksio` 包），还没到发请求那一步，所以 `NO_PROXY` 根本没机会被检查——它是**请求阶段**生效的，不是构造阶段。

正确组合：

1. `pip install "httpx[socks]"` → httpx 能成功创建 SOCKS transport（构造阶段不再爆炸）；
2. `NO_PROXY` 白名单 → 发请求时检查命中国内域名 → 走直连，不绕代理。

国内 LLM 服务不该走代理（既慢又可能被识别成异常 IP），所以两步都要。

---

## 4. 永久 NO_PROXY 设置（已完成）

修改了 `~/.bashrc:129`，把 `no_proxy` 扩展为以下白名单，并同步导出大写 `NO_PROXY`：

```bash
export no_proxy="localhost,127.0.0.1,::1,bigmodel.cn,open.bigmodel.cn,siliconflow.cn,api.siliconflow.cn,aliyuncs.com,dashscope.aliyuncs.com,deepseek.com,api.deepseek.com,zhipuai.cn"
export NO_PROXY="$no_proxy"
```

修改前已自动备份：`~/.bashrc.bak.<时间戳>`。

**为什么不影响 Windows 代理镜像**：
- `NO_PROXY` 只是告诉 HTTP 客户端（httpx/curl/requests）"访问这些域名时跳过代理直连"；
- 它**不修改** `http_proxy/https_proxy/all_proxy` 三个变量本身，海外服务仍走 Clash；
- 它**不碰** `/etc/wsl.conf` 或 `.wslconfig` 中的 networkingMode/mirrored 设置——那是 OS 路由层；
- Windows 系统代理设置完全不受影响。

简单说：仅仅多了几个国内域名走直连，其他流量照旧。

---

## 5. 让设置生效

修改 `~/.bashrc` 后**需要新开终端**或：

```bash
source ~/.bashrc
mamba activate tsci   # mamba 激活在 source 后会被重置
```

验证：

```bash
echo "$NO_PROXY"      # 应包含 bigmodel.cn 等
echo "$all_proxy"     # 应保留 socks5://127.0.0.1:7890
```

---

## 6. 临时一次性绕过代理（备用）

如果不想动 `.bashrc`，临时跑某条命令：

```bash
NO_PROXY=open.bigmodel.cn python 01_curator_minimal.py
# 或
unset all_proxy && python 01_curator_minimal.py
```

---

## 7. 添加新 LLM 服务的 SOP

将来接入新的国内 LLM 服务（比如月之暗面 / 百川 / Minimax），需要：

1. 在 `01_curator_minimal.py` 的 `PROVIDERS` 字典里加一项；
2. 在 `.env.example` 里加对应 KEY；
3. **同步把新域名加到 `~/.bashrc:129` 的 `no_proxy`**，否则会被 SOCKS proxy 卡住。

---

## 8. 故障速查

| 现象 | 处理 |
|---|---|
| `ImportError: socksio not installed` | `pip install "httpx[socks]"`（NO_PROXY 单独修不好，见 §3.1） |
| 海外服务（GitHub / PyPI）连不上 | 检查 `all_proxy` 是否还在；`curl -v https://github.com` 看是否走代理 |
| `mamba activate` 报错 | `source ~/Downloads/ENTER/etc/profile.d/conda.sh` |
| pip 装包慢 | 加镜像：`pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple`（通过代理也行，看哪个快） |
