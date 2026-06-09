#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import sys
import os
import time
import signal

def main():
    print("正在启动 Ollama 后端服务...")
    # 启动 FastAPI 后端（监听 8000 端口）
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "fastapi_backend:app", "--host", "127.0.0.1", "--port", "8000"],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
    )
    time.sleep(2)  # 等待后端启动

    print("正在启动 Streamlit 前端...")
    # 启动 Streamlit 应用（监听 8080 端口）
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "db.py", "--server.port", "8080"],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
    )

    print("所有服务已启动。按 Ctrl+C 终止所有服务。")
    try:
        # 等待两个进程结束
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n正在终止服务...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("服务已终止。")

if __name__ == "__main__":
    main()