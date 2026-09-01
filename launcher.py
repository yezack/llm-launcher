#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Launcher — beellama / llama-server 便携启动器
功能: 配置切换启动 / 停止卸载模型 / 最小化到托盘 / 退出自动卸载
依赖: tkinter(标准库), pystray + Pillow(托盘)
"""

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import functools
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False

# ---------------------------------------------------------------- 路径常量
APP_DIR = os.path.dirname(os.path.abspath(
    sys.executable if getattr(sys, "frozen", False) else __file__))
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
LOG_DIR = os.path.join(APP_DIR, "logs")
DEFAULT_SERVER = os.path.join(APP_DIR, "bin", "llama-server.exe")


def resource_path(name):
    """打包(frozen)与源码运行时的资源路径。"""
    return os.path.join(getattr(sys, "_MEIPASS", APP_DIR), name)


ICON_PATH = resource_path("icon.ico")

# 保险丝代理请求体上限(64MB): 防超大 body 一次性读入内存
MAX_BODY_BYTES = 64 * 1024 * 1024
# 日志保留: GUI 日志区最多行数; logs/ 目录最多保留文件数
LOG_GUI_MAX_LINES = 500
LOG_DIR_MAX_FILES = 500

# ---------------------------------------------------------------- 参数映射
# 配置 JSON 里的键 -> (命令行参数, 转换函数)
ARGS_MAP = {
    "host":              ("--host", str),
    "port":              ("--port", int),
    "ctx":               ("-c", int),
    "ngl":               ("-ngl", int),
    "threads":           ("-t", int),
    "parallel":          ("-np", int),
    "fa":                ("-fa", str),
    "cache_type_k":      ("-ctk", str),
    "cache_type_v":      ("-ctv", str),
    "kv_tail_tokens":    ("--kv-tail-tokens", str),
    "spec_type":         ("--spec-type", str),
    "spec_draft_model":  ("--spec-draft-model", str),
    "spec_draft_n_max":  ("--spec-draft-n-max", int),
    "reasoning":         ("--reasoning", str),
    "reasoning_effort":  ("--reasoning-effort", str),
    "predict":           ("-n", int),
    "image_min_tokens":  ("--image-min-tokens", int),
    "batch":             ("-b", int),
    "ubatch":            ("-ub", int),
    "slot_context":      ("--slot-context", int),
    "cache_ram":         ("--cache-ram", int),
}


def expand_path(p):
    """展开环境变量(%VAR%/$VAR)与 ~。"""
    if not p:
        return p
    p = os.path.expanduser(p)
    p = os.path.expandvars(p)
    return p


def resolve_path(p, base_dir=None):
    """解析模型/mmproj 路径: 支持绝对路径、环境变量、相对 base_dir 的路径。"""
    if not p:
        return p
    p = expand_path(p)
    if os.path.isabs(p):
        return os.path.normpath(p)
    if base_dir:
        return os.path.normpath(os.path.join(expand_path(base_dir), p))
    return os.path.normpath(p)


def load_config(path=CONFIG_PATH):
    """加载主配置 + config.d/ 下每个模型一个文件, 合并为 models 数组。
    主配置只放顶层(host/port/env 等) + models 索引; 每个模型文件独立, 改错一个不影响其他。
    缺失/损坏的模型文件不静默跳过, 警告写入 cfg['_warnings'] 供 UI 提示。"""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(path))
    models = []
    warnings = []
    for ref in cfg.get("models", []):
        fname = ref.get("file", "")
        if not fname:
            warnings.append("models 索引项缺少 file 字段: %r" % (ref,))
            continue
        fpath = fname if os.path.isabs(fname) else os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            warnings.append("模型配置缺失: %s" % fpath)
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                m = json.load(f)
        except Exception as e:
            warnings.append("模型配置解析失败: %s (%s)" % (fpath, e))
            continue
        m.setdefault("server_executable", cfg.get("server_executable"))
        models.append(m)
    cfg["models"] = models
    cfg["_warnings"] = warnings
    return cfg


def build_cmd(profile, server_exe, defaults=None, base_dir=None):
    """由 profile 生成 llama-server 命令行参数列表。
    defaults 提供全局默认参数(如 host/port), profile 的 args 可覆盖。
    base_dir 为 model/mmproj 相对路径的根目录(可用环境变量)。"""
    model = resolve_path(profile.get("model", ""), base_dir)
    if not model:
        raise ValueError("配置缺少 model 字段")
    if not os.path.exists(model):
        raise FileNotFoundError("模型文件不存在: %s" % model)

    cmd = [server_exe, "-m", model]

    mmproj = resolve_path(profile.get("mmproj"), base_dir)
    if mmproj:
        if not os.path.exists(mmproj):
            raise FileNotFoundError("mmproj 文件不存在: %s" % mmproj)
        cmd += ["--mmproj", mmproj]

    alias = profile.get("alias")
    if alias:
        cmd += ["--alias", str(alias)]

    args = dict(defaults or {})
    args.update(profile.get("args") or {})
    for key, (flag, conv) in ARGS_MAP.items():
        val = args.get(key)
        if val is None or val == "":
            continue
        if isinstance(val, bool):
            val = "on" if val else "off"
        cmd += [flag, str(conv(val))]
    return cmd


def port_in_use(port):
    """检查端口是否被占用。非法端口视为占用(安全失败)。"""
    try:
        port = int(port)
        if not (0 <= port <= 65535):
            return True
    except (TypeError, ValueError):
        return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def find_free_port(start=18000):
    """从 start 起探测第一个空闲端口(供内部端口使用)。"""
    for port in range(start, 65535):
        if not port_in_use(port):
            return port
    return 18000


# ---------------------------------------------------------------- 输入保险丝(前置代理)
class PromptGuardProxy:
    """前置代理: 校验单次输入长度, 超限直接拒绝(不碰GPU), 否则转发到 llama-server。"""

    def __init__(self, listen_port, upstream_port, max_tokens, log_cb, listen_host="0.0.0.0"):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.upstream_port = upstream_port
        self.max_tokens = max_tokens
        self.log = log_cb
        self.httpd = None
        self.thread = None

    def start(self):
        if self.max_tokens <= 0:
            return False
        try:
            self.httpd = _GuardServer((self.listen_host, self.listen_port), self)
        except OSError as e:
            self.log("[保险丝] 端口 %s 绑定失败: %s" % (self.listen_port, e))
            return False
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.log("[保险丝] 已启用: 单次输入估算超过 %s tokens 将被拦截" % self.max_tokens)
        return True

    def stop(self):
        if self.httpd:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
            self.httpd = None


class _GuardServer(ThreadingHTTPServer):
    def __init__(self, addr, guard):
        self.guard = guard
        super().__init__(addr, _GuardHandler)


class _GuardHandler(BaseHTTPRequestHandler):
    """转发 + 预检。"""

    def log_message(self, *args):
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        guard = self.server.guard
        length = int(self.headers.get("Content-Length") or 0)
        # --- 请求体大小上限: 防超大 body 吃内存 ---
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": {
                "code": 413,
                "message": "request body too large: %d bytes, limit is %d" % (length, MAX_BODY_BYTES),
                "type": "payload_too_large",
            }})
            return
        body = self.rfile.read(length) if length else b""

        # --- 单次输入长度预检 ---
        if guard.max_tokens > 0 and self.path in ("/v1/completions", "/v1/chat/completions"):
            est = 0
            try:
                data = json.loads(body)
                if "prompt" in data:
                    p = data["prompt"]
                    text = p if isinstance(p, str) else json.dumps(p)
                elif "messages" in data:
                    text = " ".join(str(m.get("content", "")) for m in data.get("messages", []))
                else:
                    text = ""
                est = max(1, len(text) // 4)  # 粗略估算 token 数(中英混合偏严)
            except Exception:
                est = 0
            if est > guard.max_tokens:
                guard.log("[保险丝] 拦截超长请求: 估算 %s tokens > 上限 %s" % (est, guard.max_tokens))
                self._send_json(400, {"error": {
                    "code": 400,
                    "message": "input too long: ~%s tokens estimated, limit is %s (请切块或缩短输入)"
                               % (est, guard.max_tokens),
                    "type": "input_too_long_error",
                }})
                return

        # --- 转发到上游 llama-server ---
        url = "http://127.0.0.1:%s%s" % (guard.upstream_port, self.path)
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Accept": self.headers.get("Accept", "*/*"),
        })
        try:
            resp = urllib.request.urlopen(req, timeout=900)
            status, headers, stream = resp.status, resp.headers, resp
        except urllib.error.HTTPError as e:
            status, headers, stream = e.code, e.headers, e
        except Exception as e:
            self._send_json(502, {"error": {"code": 502, "message": "upstream error: %s" % e,
                                            "type": "upstream_error"}})
            return

        # 流式转发(SSE): 逐块透传
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() in ("content-length", "transfer-encoding", "connection", "keep-alive"):
                continue
            self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def do_GET(self):
        """转发 GET 请求(如 /health、/v1/models)到上游。"""
        guard = self.server.guard
        url = "http://127.0.0.1:%s%s" % (guard.upstream_port, self.path)
        req = urllib.request.Request(url, headers={
            "Accept": self.headers.get("Accept", "*/*"),
        })
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            status, headers, body = resp.status, resp.headers, resp.read()
            resp.close()
        except urllib.error.HTTPError as e:
            status, headers, body = e.code, e.headers, e.read()
        except Exception as e:
            self._send_json(502, {"error": {"code": 502, "message": "upstream error: %s" % e,
                                            "type": "upstream_error"}})
            return
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() in ("content-length", "transfer-encoding", "connection", "keep-alive"):
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------- 服务管理
class ServerManager:
    """管理 llama-server 子进程: 启动 / 停止 / 日志输出。"""

    def __init__(self):
        self.proc = None
        self.log_queue = queue.Queue()
        self._log_file = None

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def pid(self):
        return self.proc.pid if self.running else None

    def start(self, cmd, env=None):
        self.stop()  # 确保先停旧的
        os.makedirs(LOG_DIR, exist_ok=True)
        self._cleanup_old_logs()
        log_path = os.path.join(LOG_DIR, time.strftime("server_%Y%m%d_%H%M%S.log"))
        self._log_file = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            env=env,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        threading.Thread(target=self._reader, daemon=True).start()
        return log_path

    def _cleanup_old_logs(self):
        """logs/ 目录只保留最近 LOG_DIR_MAX_FILES 个 server_*.log, 删最旧。"""
        try:
            files = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR)
                     if f.startswith("server_") and f.endswith(".log")]
            if len(files) <= LOG_DIR_MAX_FILES:
                return
            files.sort(key=os.path.getmtime, reverse=True)
            for old in files[LOG_DIR_MAX_FILES:]:
                try:
                    os.remove(old)
                except Exception:
                    pass
        except Exception:
            pass

    def _reader(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            self.log_queue.put(line)
            if self._log_file and not self._log_file.closed:
                try:
                    self._log_file.write(line + "\n")
                    self._log_file.flush()
                except Exception:
                    pass

    def stop(self):
        """停止服务并卸载模型(释放显存)。Windows 下杀整个进程树。"""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                    capture_output=True, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            try:
                self.proc.wait(timeout=8)
            except Exception:
                pass
        if self._log_file and not self._log_file.closed:
            try:
                self._log_file.close()
            except Exception:
                pass
        self.proc = None

    def drain_logs(self, max_lines=200):
        """取回日志队列内容(供 GUI 轮询)。"""
        lines = []
        try:
            while True:
                lines.append(self.log_queue.get_nowait())
                if len(lines) >= max_lines:
                    break
        except queue.Empty:
            pass
        return lines


# ---------------------------------------------------------------- GUI
class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.cfg = None
        self.server = ServerManager()
        self.tray = None
        self.tray_thread = None
        self.exiting = False
        self.proxy = None
        self._server_port = None
        self._last_decode_total = None
        self._last_tps_t = None

        self.root.title("LLM Launcher")
        self.root.geometry("860x560")
        self.root.minsize(640, 420)
        # 窗口 / 任务栏图标
        if os.path.exists(ICON_PATH):
            try:
                self.root.iconbitmap(ICON_PATH)
            except Exception:
                pass

        self._build_ui()
        self.reload_config()
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)
        self.root.after(200, self._poll_loop)
        self.root.after(1000, self._sample_speed)
        self._setup_tray()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="模型:").pack(side="left")
        self.model_combo = ttk.Combobox(top, state="readonly", width=26)
        self.model_combo.pack(side="left", padx=6)
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self._on_model_changed())

        ttk.Label(top, text="配置:").pack(side="left")
        self.profile_combo = ttk.Combobox(top, state="readonly", width=34)
        self.profile_combo.pack(side="left", padx=6)
        self.profile_combo.bind("<<ComboboxSelected>>", lambda e: self._on_profile_changed())

        ttk.Label(top, text="思考:").pack(side="left")
        self.thinking_combo = ttk.Combobox(top, state="readonly", width=10,
                                           values=["关闭", "low", "medium", "high", "xhigh"])
        self.thinking_combo.set("")
        self.thinking_combo.pack(side="left", padx=6)

        btn_row = ttk.Frame(self.root)
        btn_row.pack(fill="x", **pad)

        self.start_btn = ttk.Button(btn_row, text="▶ 启动", command=self.start_server)
        self.start_btn.pack(side="left", padx=2)
        self.stop_btn = ttk.Button(btn_row, text="■ 停止 / 卸载模型", command=self.stop_server)
        self.stop_btn.pack(side="left", padx=2)
        ttk.Button(btn_row, text="↻ 重新加载配置", command=self.reload_config).pack(side="left", padx=2)
        ttk.Button(btn_row, text="✎ 编辑配置文件", command=self.open_config).pack(side="left", padx=2)
        ttk.Button(btn_row, text="📂 打开日志目录", command=self.open_logs).pack(side="left", padx=2)
        if HAS_TRAY:
            ttk.Button(btn_row, text="— 最小化到托盘", command=self.minimize_to_tray).pack(side="right", padx=2)

        desc = ttk.LabelFrame(self.root, text="配置说明")
        desc.pack(fill="x", **pad)
        self.desc_label = ttk.Label(desc, text="", wraplength=820, justify="left")
        self.desc_label.pack(fill="x", padx=8, pady=4)

        status = ttk.LabelFrame(self.root, text="服务状态")
        status.pack(fill="x", **pad)
        self.status_label = ttk.Label(status, text="● 未运行", foreground="gray")
        self.status_label.pack(side="left", padx=8, pady=4)
        self.pid_label = ttk.Label(status, text="PID: —")
        self.pid_label.pack(side="left", padx=8)
        self.listen_label = ttk.Label(status, text="监听: —")
        self.listen_label.pack(side="left", padx=8)
        self.alias_label = ttk.Label(status, text="调用名: —")
        self.alias_label.pack(side="left", padx=8)
        self.speed_label = ttk.Label(status, text="吞吐: —")
        self.speed_label.pack(side="right", padx=8)

        log_frame = ttk.LabelFrame(self.root, text="服务日志")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=14, state="disabled",
                                                  font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_text.tag_configure("err", foreground="#c62828")
        self.log_text.tag_configure("ok", foreground="#2e7d32")

        tip = ttk.Label(self.root, text="提示: 关闭窗口将最小化到托盘; 托盘菜单「退出」会停止服务并卸载模型。",
                        foreground="gray")
        tip.pack(anchor="w", padx=8, pady=(0, 6))

    # ---------------- 配置 ----------------
    def reload_config(self):
        try:
            self.cfg = load_config()
        except Exception as e:
            messagebox.showerror("配置错误", "无法加载配置文件:\n%s\n\n路径: %s" % (e, CONFIG_PATH))
            self.cfg = {"server_executable": "bin/llama-server.exe", "models": []}
            return
        models = self.cfg.get("models", [])
        self.model_combo["values"] = [m.get("name", "未命名模型") for m in models]
        self.profile_combo["values"] = []
        self.profile_combo.set("")
        if models:
            self.model_combo.current(0)
            self._update_profiles()
            self._show_profile_desc()
        warnings = self.cfg.get("_warnings") or []
        if warnings:
            messagebox.showwarning("配置警告", "\n".join(warnings))

    def current_model(self):
        idx = self.model_combo.current()
        models = self.cfg.get("models", [])
        if idx < 0 or idx >= len(models):
            return None
        return models[idx]

    def _on_model_changed(self):
        self._update_profiles()
        self._show_profile_desc()

    def _on_profile_changed(self):
        self._sync_thinking()
        self._show_profile_desc()

    def _update_profiles(self):
        m = self.current_model()
        profiles = (m or {}).get("profiles", [])
        self.profile_combo["values"] = [p.get("name", "未命名配置") for p in profiles]
        if profiles:
            self.profile_combo.current(0)
        self._sync_thinking()

    def _sync_thinking(self):
        """切换配置时, 把「思考」下拉框同步为当前配置的档位(可手动改覆盖, 不保存)。"""
        m = self.current_model()
        p = self.current_profile()
        if not p:
            self.thinking_combo.set("")
            return
        args = dict(m.get("args") or {})
        args.update(p.get("args") or {})
        if str(args.get("reasoning", "")).lower() == "off":
            self.thinking_combo.set("关闭")
            return
        eff = args.get("reasoning_effort")
        values = list(self.thinking_combo["values"])
        if eff in values:
            self.thinking_combo.set(eff)
        else:
            self.thinking_combo.set("")

    def current_profile(self):
        m = self.current_model()
        if not m:
            return None
        idx = self.profile_combo.current()
        profiles = m.get("profiles", [])
        if idx < 0 or idx >= len(profiles):
            return None
        return profiles[idx]

    def _show_profile_desc(self):
        p = self.current_profile()
        m = self.current_model()
        if not p or not m:
            self.desc_label.config(text="")
            return
        desc = p.get("description", "")
        merged = self.merged_args()
        args = dict(merged)
        alias = p.get("alias") or m.get("alias") or "—"
        host = merged.get("host", "127.0.0.1")
        port = merged.get("port", "—")
        model_name = m.get("name", "")
        extra = " | ".join("%s=%s" % (k, v) for k, v in sorted(args.items()))
        self.desc_label.config(text="%s\n[模型] %s   [调用名] %s   [监听] %s:%s\n[参数] %s" % (desc, model_name, alias, host, port, extra))
        self._update_status()

    # ---------------- 服务控制 ----------------
    def server_executable(self):
        """解析引擎可执行文件路径。优先 profile → model → 全局默认。"""
        p = self.current_profile()
        m = self.current_model()
        exe = ((p or {}).get("server_executable")
               or (m or {}).get("server_executable")
               or self.cfg.get("server_executable") or "bin/llama-server.exe")
        exe = expand_path(exe)
        if not os.path.isabs(exe):
            exe = os.path.join(APP_DIR, exe)
        return exe

    def merged_args(self):
        """全局默认参数(host/port) + model 公共 args + profile args(覆盖)。"""
        base = {
            "host": self.cfg.get("host", "127.0.0.1"),
            "port": self.cfg.get("port", 62345),
        }
        m = self.current_model()
        if m:
            base.update(m.get("args") or {})
        p = self.current_profile()
        if p:
            base.update(p.get("args") or {})
        return base

    def start_server(self):
        m = self.current_model()
        p = self.current_profile()
        if not p or not m:
            messagebox.showwarning("提示", "请先选择模型和配置")
            return
        try:
            exe = self.server_executable()
            if not os.path.exists(exe):
                raise FileNotFoundError("找不到 llama-server.exe: %s" % exe)
            merged = self.merged_args()
            port = int(merged.get("port") or 62345)
            external_host = str(merged.get("host") or "0.0.0.0")
            # 有效配置 = model 公共字段(模型路径/mmproj/alias) + profile args(覆盖)
            eff = dict(p)
            eff.setdefault("model", m.get("model"))
            eff.setdefault("mmproj", m.get("mmproj"))
            eff.setdefault("alias", m.get("alias"))
            eff["args"] = dict(p.get("args") or {})
            # 手动思考强度覆盖(「思考」下拉框): 覆盖配置默认, 仅本次启动生效
            thinking = self.thinking_combo.get()
            if thinking == "关闭":
                eff["args"]["reasoning"] = "off"
                eff["args"].pop("reasoning_effort", None)
            elif thinking in ("low", "medium", "high", "xhigh"):
                eff["args"]["reasoning"] = "on"
                eff["args"]["reasoning_effort"] = thinking
            use_guard = bool(self.cfg.get("use_guard", True))
            if use_guard:
                # 代理模式: llama-server 绑本机内部端口, 代理绑对外地址
                try:
                    cfg_internal = int(self.cfg.get("internal_port") or 0)
                except (TypeError, ValueError):
                    cfg_internal = 0
                if cfg_internal and not port_in_use(cfg_internal):
                    internal_port = cfg_internal
                else:
                    internal_port = find_free_port()
                cmd = build_cmd(eff, exe, dict(merged, host="127.0.0.1", port=internal_port), base_dir=self.cfg.get("base_dir"))
            else:
                # 直连模式: llama-server 直接绑对外地址, 单端口, 无保险丝
                internal_port = None
                cmd = build_cmd(eff, exe, merged, base_dir=self.cfg.get("base_dir"))
            cmd.append("--metrics")  # GUI 实时吞吐依赖 /metrics 端点
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            return

        if port_in_use(port):
            if not messagebox.askyesno("端口被占用",
                                       "端口 %s 已被占用, 可能已有服务在运行。\n仍要启动吗?"
                                       "(会导致新进程启动失败)" % port):
                return

        try:
            env = os.environ.copy()
            env.update(self.cfg.get("env") or {})
            env.update(m.get("env") or {})
            env.update(p.get("env") or {})
            log_path = self.server.start(cmd, env=env)
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            return
        # 输入保险丝代理: llama-server 绑定本机内部端口, 代理绑对外地址
        max_tokens = int(self.cfg.get("max_prompt_tokens") or 0)
        self.proxy = None
        if use_guard and max_tokens > 0 and not port_in_use(internal_port):
            self.proxy = PromptGuardProxy(port, internal_port, max_tokens, self._safe_log,
                                          listen_host=external_host)
            self.proxy.start()
        self._append_log(">>> 启动命令: %s" % " ".join(cmd), "ok")
        t_on = eff["args"].get("reasoning", "auto")
        t_eff = eff["args"].get("reasoning_effort") or "默认"
        self._append_log(">>> 思考: %s / 强度: %s" % (t_on, t_eff), "ok")
        self._append_log(">>> 日志文件: %s" % log_path, "ok")
        # 记录 llama-server 实际端口(供停止时查忙闲): guard 模式=内部端口, 直连=对外端口
        self._server_port = internal_port if use_guard else port
        if self.proxy:
            self._append_log(">>> 对外: %s:%s (代理+保险丝) → 内部 127.0.0.1:%s" % (
                external_host, port, internal_port), "ok")
        else:
            self._append_log(">>> 直连模式: %s:%s (无代理, 单端口)" % (external_host, port), "ok")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._update_status()

    def _slots_busy(self):
        """查询 llama-server /slots, 是否有请求正在生成(防止 GPU 忙时强杀)。查不到视为不忙。"""
        port = getattr(self, "_server_port", None)
        if not port:
            return False
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/slots" % port, timeout=3) as r:
                slots = json.loads(r.read())
            return any(str(s.get("state", "")).lower() in ("processing", "busy")
                       for s in slots)
        except Exception:
            return False

    def stop_server(self):
        if self.server.running:
            self._append_log(">>> 正在停止服务并卸载模型(释放显存)...")
            # 优雅停止: 若有请求正在生成, 先确认再强杀(避免 GPU 忙时崩溃)
            if self._slots_busy():
                if not messagebox.askyesno(
                        "正在生成中",
                        "有请求正在生成, 立即停止会中断生成并强杀进程(可能影响显存)。\n"
                        "等生成完成再停? 选「否」= 现在强制停止。"):
                    self._append_log(">>> 已取消停止, 等待生成完成", "ok")
                    return
                self._append_log(">>> 用户确认强制停止", "err")
        self._stop_proxy()
        self.server.stop()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._append_log(">>> 服务已停止, 模型已卸载", "ok")
        self._update_status()

    def _stop_proxy(self):
        if self.proxy:
            try:
                self.proxy.stop()
            except Exception:
                pass
            self.proxy = None

    # ---------------- 日志 / 状态 ----------------
    def _append_log(self, text, tag=None):
        self.log_text.config(state="normal")
        stamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", "[%s] %s\n" % (stamp, text), tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _safe_log(self, text, tag=None):
        """线程安全日志: 从任意线程调度到主线程写 tkinter(供 guard 代理回调)。"""
        try:
            self.root.after(0, lambda: self._append_log(text, tag))
        except Exception:
            pass

    def _poll_loop(self):
        for line in self.server.drain_logs():
            low = line.lower()
            tag = "err" if ("error" in low or "illegal" in low or "failed" in low) else None
            self._append_log(line, tag)
            # 日志上限: 超过 LOG_GUI_MAX_LINES 行时裁掉最旧的 100 行
            try:
                line_count = int(self.log_text.index("end-1c").split(".")[0])
            except Exception:
                line_count = 0
            if line_count > LOG_GUI_MAX_LINES:
                self.log_text.config(state="normal")
                self.log_text.delete("1.0", "100.0")
                self.log_text.config(state="disabled")
        self._update_status()
        self.root.after(200, self._poll_loop)

    def _sample_speed(self):
        """每秒采样 /metrics: 用 n_decode(+投机接受数) 差值算实时总吞吐(多路求和)。
        无投机时 n_decode 增量 = 生成 token 数; 投机时加上接受的 draft tokens 更接近真实输出。"""
        port = getattr(self, "_server_port", None)
        if not port or not self.server.running:
            self.speed_label.config(text="吞吐: —")
            self._last_decode_total = None
            self.root.after(1000, self._sample_speed)
            return
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/metrics" % port, timeout=3) as r:
                text = r.read().decode("utf-8", "replace")
            decode = self._metric_value(text, "llamacpp:n_decode_total")
            accepted = self._metric_value(text, "llamacpp:spec_decode_num_accepted_tokens_total")
            if decode is not None:
                now = time.time()
                if self._last_decode_total is not None and self._last_tps_t is not None:
                    dt = now - self._last_tps_t
                    delta = (decode + (accepted or 0)) - self._last_decode_total
                    if dt > 0 and delta >= 0:
                        self.speed_label.config(text="吞吐: %.1f tok/s" % (delta / dt))
                self._last_decode_total = decode + (accepted or 0)
                self._last_tps_t = now
        except Exception:
            self.speed_label.config(text="吞吐: —")
            self._last_decode_total = None
        self.root.after(1000, self._sample_speed)

    @staticmethod
    def _metric_value(text, name):
        """从 prometheus 文本取指定指标的值(找不到返回 None)。"""
        for line in text.splitlines():
            if line.startswith(name + " "):
                try:
                    return float(line.split()[-1])
                except ValueError:
                    return None
        return None

    def _update_status(self):
        p = self.current_profile()
        m = self.current_model()
        merged = self.merged_args()
        host = merged.get("host", "127.0.0.1")
        port = merged.get("port", "—")
        alias = (p.get("alias") or (m or {}).get("alias") or "—") if p else "—"
        if self.server.running:
            self.status_label.config(text="● 运行中", foreground="#2e7d32")
            self.pid_label.config(text="PID: %s" % self.server.pid)
        else:
            self.status_label.config(text="● 未运行", foreground="gray")
            self.pid_label.config(text="PID: —")
        self.listen_label.config(text="监听: %s:%s" % (host, port))
        self.alias_label.config(text="调用名: %s" % alias)

    # ---------------- 托盘 ----------------
    def _setup_tray(self):
        if not HAS_TRAY:
            return
        img = None
        if os.path.exists(ICON_PATH):
            try:
                img = Image.open(ICON_PATH)
            except Exception:
                img = None
        if img is None:
            img = self._tray_image()
        menu = pystray.Menu(
            pystray.MenuItem("显示窗口", self._tray_show, default=True),
            pystray.MenuItem("启动服务", self._tray_start),
            pystray.MenuItem("停止/卸载", self._tray_stop),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出(停止服务)", self._tray_quit),
        )
        self.tray = pystray.Icon("llm-launcher", img, "LLM Launcher", menu)
        self.tray_thread = threading.Thread(target=self.tray.run, daemon=True)
        self.tray_thread.start()

    @staticmethod
    def _tray_image():
        img = Image.new("RGB", (64, 64), "#1e88e5")
        d = ImageDraw.Draw(img)
        d.rectangle([12, 12, 52, 52], fill="#ffffff")
        d.polygon([(20, 40), (32, 20), (44, 40)], fill="#1e88e5")
        return img

    def _tray_show(self, icon=None, item=None):
        self.root.after(0, self._show_from_tray)

    def _show_from_tray(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _tray_start(self, icon=None, item=None):
        self.root.after(0, self.start_server)

    def _tray_stop(self, icon=None, item=None):
        self.root.after(0, self.stop_server)

    def _tray_quit(self, icon=None, item=None):
        self.root.after(0, self.quit_app)

    def minimize_to_tray(self):
        if HAS_TRAY and self.tray:
            self.root.withdraw()
            if self.tray_thread and self.tray_thread.is_alive():
                self.tray.notify("已最小化到托盘, 服务仍在运行" if self.server.running
                                 else "已最小化到托盘", "LLM Launcher")

    def on_window_close(self):
        """点窗口 X: 有托盘则最小化到托盘, 否则退出。"""
        if HAS_TRAY and self.tray:
            self.minimize_to_tray()
        else:
            self.quit_app()

    # ---------------- 退出 ----------------
    def quit_app(self):
        if self.exiting:
            return
        self.exiting = True
        if self.server.running:
            self._append_log(">>> 程序退出, 正在停止服务并卸载模型...")
        self._stop_proxy()
        self.server.stop()
        if self.tray and self.tray_thread and self.tray_thread.is_alive():
            try:
                self.tray.stop()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass
        # 不再 os._exit: 让 mainloop 自然结束, daemon 线程(托盘/日志读取)随进程退出。
        # main() 里的 atexit 兜底会再调用一次 server.stop(), 确保显存释放。

    # ---------------- 辅助 ----------------
    def open_config(self):
        subprocess.Popen(["notepad", CONFIG_PATH],
                         creationflags=subprocess.CREATE_NO_WINDOW)

    def open_logs(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        subprocess.Popen(["explorer", LOG_DIR],
                         creationflags=subprocess.CREATE_NO_WINDOW)


def main():
    root = tk.Tk()
    app = LauncherApp(root)
    # 兜底: 异常退出时也卸载模型
    import atexit
    atexit.register(lambda: app.server.stop())
    root.mainloop()
    app.server.stop()


if __name__ == "__main__":
    main()
