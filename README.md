# LLM Launcher

Windows 托盘版本地大模型启动器，基于 **llama.cpp**。多配置预设一键切换、启动 / 停止（自动释放显存）、实时日志查看、托盘常驻、原生多对话（slot）支持。

## 功能特性

- **llama.cpp 引擎**：GGUF 模型，便携单文件，个人电脑友好
- **两级配置**：先选模型，再选该模型下的配置预设（profile），互不干扰
- **多文件配置**：每个模型一个 `config.d/*.json`，改错一个不影响其他模型
- **一键启停**：停止时杀整个进程树，释放显存
- **托盘常驻**：关闭窗口最小化到托盘，托盘菜单可启动 / 停止 / 退出
- **多对话切换**：`-np` 多 slot + `slot_id` 原生支持，多个对话互不干扰
- **KV cache 量化**：`q4_0`/`q8_0` 等，长上下文省显存
- **投机解码**：MTP（`--spec-type draft-mtp`）或 DFlash2（`draft-dflash` + 独立 drafter）

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
copy config.example.json config.json
mkdir config.d & copy config.d\example-model.json config.d\my-model.json   # 改成自己的模型
python launcher.py
```

### 2. 打包为 exe（免 Python）

```bat
build.bat
```

产物在 `dist\LLMLauncher\LLMLauncher.exe`。

## 配置说明

配置分两层：**主配置** `config.json`（公共设置 + 模型索引）+ **模型文件** `config.d/*.json`（每个模型一个文件，含该模型下的多个 profile）。

### 主配置 `config.json`

```json
{
  "base_dir": "D:/path/to/models",
  "server_executable": "bin/llama-server.exe",
  "host": "0.0.0.0",
  "port": 8080,
  "use_guard": false,
  "env": {},
  "models": [
    { "file": "config.d/example-model.json" },
    { "file": "config.d/another-model.json" }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `base_dir` | 模型根目录，模型文件里的相对路径以此为基准 |
| `server_executable` | `llama-server.exe` 路径（相对程序目录或绝对）。模型文件可单独覆盖（版本切换） |
| `host` / `port` | 监听地址 / 端口 |
| `use_guard` | 是否启用"输入保险丝"代理（超长输入拦截） |
| `env` | 全局环境变量（如 `LLAMA_ATTN_ROT_DISABLE=1`） |
| `models` | 模型索引，每项 `{"file": "config.d/xxx.json"}` |

### 模型文件 `config.d/*.json`

一个模型一个文件，GUI 第一个下拉框选模型，第二个下拉框选该模型下的 profile：

```json
{
  "name": "示例模型 · Q4_K_M",
  "alias": "example-model",
  "model": "D:/models/model-Q4_K_M.gguf",
  "mmproj": "",
  "env": {},
  "profiles": [
    {
      "name": "64K 单对话 · MTP",
      "description": "Q4 权重 + K4V4 KV + MTP 投机解码。64K 单对话。",
      "alias": "example-model-64k",
      "args": {
        "ctx": 65536,
        "ngl": 99,
        "threads": 16,
        "parallel": 1,
        "fa": "on",
        "cache_type_k": "q4_0",
        "cache_type_v": "q4_0",
        "spec_type": "draft-mtp",
        "spec_draft_n_max": 4
      },
      "env": {}
    }
  ]
}
```

模型级字段（公共，所有 profile 继承）：`name`、`alias`、`model`、`mmproj`、`env`、`server_executable`。
profile 级字段（可覆盖模型级）：`name`、`alias`、`description`、`args`（覆盖合并模型级 args）、`env`。

常用参数（`args`）：

| 键 | llama.cpp 参数 | 说明 |
|---|---|---|
| `ctx` | `-c` | 上下文长度（总 KV 池） |
| `ngl` | `-ngl` | GPU 层数（99 = 全层） |
| `parallel` | `-np` | 并行对话槽位数（多对话切换用） |
| `fa` | `-fa` | Flash Attention（on/off） |
| `cache_type_k` / `cache_type_v` | `-ctk` / `-ctv` | KV cache 量化（`q4_0`/`q8_0` 等） |
| `spec_type` | `--spec-type` | 投机解码类型（`draft-mtp` / `draft-dflash`） |
| `spec_draft_model` | `--spec-draft-model` | DFlash2 专用：独立 drafter 的 GGUF 路径 |
| `spec_draft_n_max` | `--spec-draft-n-max` | draft 数量（MTP 4 为实测最佳） |

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
├── config.example.json   # 主配置模板
├── config.d/             # 每个模型一个配置（example-model.json 为模板）
├── config.json           # 本地主配置（gitignore）
├── build.bat             # 打包脚本
├── requirements.txt
├── bin/                  # llama.cpp 运行时（gitignore，需自行下载）
└── logs/                 # 运行日志（gitignore）
```

## License

[MIT](LICENSE)
