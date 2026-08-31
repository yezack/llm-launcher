# LLM Launcher

Windows 托盘版本地大模型启动器，基于 **llama.cpp**。多配置预设一键切换、启动 / 停止（自动释放显存）、实时日志查看、托盘常驻、原生多对话（slot）支持。

## 功能特性

- **llama.cpp 引擎**：GGUF 模型，便携单文件，个人电脑友好
- **多预设切换**：下拉框切换不同上下文 / 对话槽位配置
- **一键启停**：停止时杀整个进程树，释放显存
- **托盘常驻**：关闭窗口最小化到托盘，托盘菜单可启动 / 停止 / 退出
- **多对话切换**：`-np` 多 slot + `slot_id` 原生支持，多个对话互不干扰
- **KV cache 量化**：`q4_0`/`q8_0` 等，长上下文省显存
- **MTP 投机解码**：`--spec-type draft-mtp` 加速生成

## 环境要求

| 项 | 说明 |
|---|---|
| 系统 | Windows 10 / 11 |
| Python | 3.11+（仅源码运行需要；打包为 exe 后无需） |
| llama.cpp | 把 `llama-server.exe` 及其 DLL 放入 `bin/`（体积大，不随仓库分发） |
| 模型 | GGUF 格式（`*.gguf`），用 `config.json` 指定路径 |

## 快速开始

### 1. 源码运行

```bash
git clone <repo-url>
cd llm-launcher
pip install -r requirements.txt
copy config.example.json config.json   # 然后按需修改模型路径
python launcher.py
```

### 2. 打包为 exe（免 Python）

```bat
build.bat
```

产物在 `dist\LLMLauncher\LLMLauncher.exe`。

## 配置说明

配置在 `config.json`（从 `config.example.json` 复制）。顶层字段：

| 字段 | 说明 |
|---|---|
| `base_dir` | 模型根目录，profile 里的相对路径以此为基准 |
| `server_executable` | `llama-server.exe` 路径（相对 `bin/` 或绝对） |
| `host` / `port` | 监听地址 / 端口 |
| `use_guard` | 是否启用"输入保险丝"代理（超长输入拦截） |
| `env` | 全局环境变量（如 `LLAMA_ATTN_ROT_DISABLE=1`） |
| `profiles` | 预设列表，GUI 下拉框选择 |

### 预设（profile）

每个 profile 一个 llama.cpp 配置：

```json
{
  "name": "Q4_K_P · 64K 单对话 · MTP",
  "alias": "model-64k",
  "model": "D:/models/model-Q4_K_M.gguf",
  "mmproj": "",
  "args": {
    "ctx": 65536,
    "ngl": 99,
    "parallel": 1,
    "fa": "on",
    "cache_type_k": "q4_0",
    "cache_type_v": "q4_0",
    "spec_type": "draft-mtp",
    "spec_draft_n_max": 4
  },
  "env": {}
}
```

常用参数（`args`）：

| 键 | llama.cpp 参数 | 说明 |
|---|---|---|
| `ctx` | `-c` | 上下文长度（总 KV 池） |
| `ngl` | `-ngl` | GPU 层数（99 = 全层） |
| `parallel` | `-np` | 并行对话槽位数（多对话切换用） |
| `fa` | `-fa` | Flash Attention（on/off） |
| `cache_type_k` / `cache_type_v` | `-ctk` / `-ctv` | KV cache 量化（`q4_0`/`q8_0` 等） |
| `spec_type` | `--spec-type` | 投机解码类型（`draft-mtp` 等） |
| `spec_draft_n_max` | `--spec-draft-n-max` | MTP draft 数量（4 为实测最佳） |

### 多对话切换（slot）

`parallel: 3` 时启动 3 个对话槽位（slot），API 请求带 `slot_id` 指定对话：

```python
# 对话 A → slot 0
{"slot_id": 0, "messages": [A的历史 + 新消息]}
# 对话 B → slot 1（来回切换，互不干扰）
{"slot_id": 1, "messages": [B的历史 + 新消息]}
```

同一 slot 的多轮对话自动复用 KV cache（每轮只 prefill 新增消息）。

## 目录结构

```
llm-launcher/
├── launcher.py           # 主程序
├── config.example.json   # 配置模板
├── config.json           # 本地配置（gitignore）
├── build.bat             # 打包脚本
├── requirements.txt
├── bin/                  # llama.cpp 运行时（gitignore，需自行下载）
└── logs/                 # 运行日志（gitignore）
```

## License

[MIT](LICENSE)
