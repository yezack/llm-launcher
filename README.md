# LLM Launcher

Windows 托盘版本地大模型启动器：统一管理 **llama.cpp** 与 **vLLM** 两种推理引擎。支持多配置预设一键切换、启动 / 停止（自动释放显存）、实时日志查看与托盘常驻。

## 功能特性

- **双引擎**：llama.cpp（GGUF，便携单文件）+ vLLM（safetensors，OpenAI 兼容 API）
- **多预设切换**：下拉框切换不同上下文 / 并发配置（例如 100K 单路、50K 双路、32K 三路）
- **一键启停**：停止时杀整个进程树，释放显存
- **托盘常驻**：关闭窗口最小化到托盘，托盘菜单可启动 / 停止 / 退出
- **vLLM agent 友好**：内置 tool-call / reasoning / auto-tool-choice 参数（Qwen3 系列）

## 环境要求

| 项 | 说明 |
|---|---|
| 系统 | Windows 10 / 11 |
| Python | 3.11+（仅源码运行需要；打包为 exe 后无需） |
| llama.cpp | 把 `llama-server.exe` 及其 DLL 放入 `bin/`（体积大，不随仓库分发） |
| vLLM | 独立的 Python venv，把解释器路径填入 `config.json` 的 `vllm.executable` |

## 快速开始

### 1. 源码运行

```bash
git clone <repo-url>
cd llm-launcher
pip install -r requirements.txt
copy config.example.json config.json   # 然后按需修改路径
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
| `server_executable` | llama.cpp 的 `llama-server.exe` 路径（相对 `bin/` 或绝对） |
| `host` / `port` | llama.cpp 默认监听地址 / 端口 |
| `vllm` | vLLM 引擎配置（见下） |
| `profiles` | 预设列表，GUI 下拉框选择 |

### vLLM 配置段

```json
"vllm": {
  "executable": "D:/path/to/vllm/venv/Scripts/python.exe",
  "module": "vllm",
  "common_args": { "host": "127.0.0.1", "port": 8000, "..." },
  "env": { "PYTHONUTF8": "1", "VLLM_HOST_IP": "127.0.0.1" }
}
```

- `executable` + `module`：通过 `python.exe -m vllm` 启动（避免 console-script 硬编码路径，便于迁移 venv）
- `common_args`：所有 vLLM 预设共享的参数（`--kv-cache-dtype`、`--quantization` 等）
- `env`：vLLM 需要的环境变量（Windows 中文环境务必 `PYTHONUTF8=1`）

### 预设（profile）

每个 profile 一个引擎：

- **llama.cpp**：`engine` 省略，`model` 指向 `.gguf` 文件，`args` 用 llama.cpp 参数（`ctx`/`ngl`/`parallel` 等）
- **vLLM**：`engine: "vllm"`，`model` 指向 safetensors 模型目录，`args` 用 vLLM 参数（`max_model_len`/`max_num_seqs` 等），可覆盖 `common_args`

## vLLM 部署提示

- 用 `python -m vllm serve <model> ...` 启动，别用 `vllm.exe`（路径迁移后 console-script 会失效）
- Windows 中文环境必须 `PYTHONUTF8=1`，否则 GBK 解码崩溃
- NVFP4 / ModelOpt 量化模型需要 `--quantization modelopt --kv-cache-dtype fp8 --trust-remote-code`

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
