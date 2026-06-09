#!/usr/bin/env python3
"""Ollama 连接诊断工具（Windows 专用）"""
import requests
import socket

def test_connection(name, url, timeout=2):
    """测试单种连接方式"""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            print(f"✅ {name}: 成功")
            return True
    except Exception as e:
        print(f"❌ {name}: 失败 - {type(e).__name__}")
    return False

print("正在诊断 Ollama 服务连接...\n")

# 测试 4 种可能的地址
test_connection("IPv4 本机", "http://127.0.0.1:11434")
test_connection("IPv6 本机", "http://[::1]:11434")
test_connection("localhost", "http://localhost:11434")
test_connection("所有接口", "http://0.0.0.0:11434")

# 检查端口是否真的在监听
print(f"\n🔍 检查端口 11434 占用情况:")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 11434))
    if result == 0:
        print("   端口 11434 正在被监听")
    else:
        print("   端口 11434 无人监听")
    sock.close()
except Exception as e:
    print(f"   检测失败: {e}")