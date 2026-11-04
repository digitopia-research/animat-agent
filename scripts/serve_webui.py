#!/usr/bin/env python3
"""
Animat Web UI 本地静态服务器。

用途：
  浏览器出于安全策略，file:// 协议下加载本地 JS 与连接 WebSocket 都可能受限，
  因此用一个极简 HTTP 服务把 webui/ 目录跑起来。

用法：
  python scripts/serve_webui.py              # 默认 0.0.0.0:8000
  python scripts/serve_webui.py 8080         # 指定端口
  ANIMAT_WEBUI_PORT=9000 python scripts/serve_webui.py

然后浏览器打开：
  http://<本机IP>:8000/

前置条件：
  MQTT broker 需开启 WebSocket 传输（默认 animat 的 mosquitto 已配置 8083 端口）。
"""

from __future__ import annotations

import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEBUI_DIR = Path(__file__).resolve().parent.parent / "webui"


class Handler(SimpleHTTPRequestHandler):
    """静态文件服务，禁用缓存便于调试。"""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # 精简日志：只看请求路径与状态码
        sys.stderr.write("  %s\n" % (fmt % args))


def main() -> None:
    if not WEBUI_DIR.is_dir():
        sys.exit(f"找不到 webui 目录: {WEBUI_DIR}")

    port = int(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.getenv("ANIMAT_WEBUI_PORT", "8000")
    )
    host = os.getenv("ANIMAT_WEBUI_HOST", "0.0.0.0")

    handler = partial(Handler, directory=str(WEBUI_DIR))
    server = ThreadingHTTPServer((host, port), handler)

    print("═" * 58)
    print("  Animat Web UI")
    print(f"  目录 : {WEBUI_DIR}")
    print(f"  地址 : http://localhost:{port}/")
    print(f"  监听 : {host}:{port}")
    print("  Ctrl+C 停止")
    print("═" * 58)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()


if __name__ == "__main__":
    main()
