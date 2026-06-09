#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ollama import Client
import json
import re

client = Client(host='http://127.0.0.1:11434')


def fetch_meta(model_name: str) -> dict:
    """安全获取元数据：template、parameters，以及嵌在 parameters 里的 system"""
    try:
        info = client.show(model_name)

        # 用 getattr 安全访问，避免 AttributeError
        template = getattr(info, 'template', '') or ''
        parameters = getattr(info, 'parameters', {}) or {}

        # system 不在 ShowResponse 顶层，而在 parameters 字典里
        system = ''
        if isinstance(parameters, dict):
            system = parameters.get('system', '')

        return {
            "template": template,
            "system": system,
            "params": json.dumps(parameters),
        }
    except Exception as e:
        # 真正的网络/服务错误才打印警告
        print(f"Warning: 无法获取 {model_name} 的元数据: {e}")
        return {"template": "", "system": "", "params": "{}"}


def classify(model_name: str, meta: dict, families: list) -> str:
    """分类逻辑不变，优先 families，其次 template"""
    fam_text = ' '.join(families).lower() if families else ""

    # P0: families 硬证据
    if 'clip' in fam_text or 'llava' in fam_text:
        return "Vision"
    if 'embedding' in fam_text or 'bert' in fam_text:
        return "Embedding"
    if 'code' in fam_text:
        return "Code"

    # P1: template 文本证据
    text = (meta["template"] + " " + meta["system"]).lower()
    if re.search(r"\bembed\b", text):
        return "Embedding"
    if any(k in text for k in ("<image>", "{{ .image }}", "llava", "bakllava")):
        return "Vision"
    if re.search(r"\b(code|completion|fill.*middle|fim)\b", text):
        return "Code"
    if re.search(r"(user|assistant|<\|im_start\|>|<\|user\|>|<\|assistant\|>)", text):
        return "LLM"

    # P2: families 兜底
    if any(f in fam_text for f in ('llama', 'qwen', 'gemma', 'mistral', 'gptoss')):
        return "LLM"

    return "Unknown"


def main():
    response = client.list()
    models = response.models
    print(models)
    if not models:
        print("未找到任何本地模型。")
        return

    result = {}
    for model_info in models:
        name = model_info.model
        details = getattr(model_info, 'details', None)
        families = getattr(details, 'families', []) if details else []

        meta = fetch_meta(name)
        result[name] = classify(name, meta, families)

    # 打印表格
    print("{:<30} {:<12}".format("MODEL", "TYPE"))
    print("-" * 42)
    for k, v in result.items():
        print("{:<30} {:<12}".format(k, v))

    # 保存 JSON
    with open("ollama_model_types.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()