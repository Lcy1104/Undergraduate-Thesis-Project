import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError

# Ollama API基础地址
OLLAMA_BASE_URL = "http://localhost:11434/api"
# 极简有效图片Base64（用于Vision能力验证）
VALID_SIMPLE_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAACXBIWXMAAAsTAAALEwEAmpwYAAAKT2lDQ1BQaG90b3Nob3AgSUNDIHByb2ZpbGUAAHjanVNnVFPpFj333vRCS4iAlEtvUhUIIFJCi4AUkSYqIQkQSoghodkVUcERRUUEG8igiAOOjoCMFVEsDIoK2AfkIaKOg6OIisr74Xuja9a89+bN/rXXPues852zzwfACAyWSDNRNYAMqUIeEeCDx8TG4eQuQIEKJHAAEAizZCFz/SMBAPh+PDwrIsAHvgABeNMLCADATZvAMByH/w/qQplcAYCEAcB0kThLCIAUAEB6jkKmAEBGAYCdmCZTAKAEAGDLY2LjAFAtAGAnf+bTAICd+Jl7AQBblCEVAaCRACATZYhEAGg7AKzPVopFAFgwABRmS8Q5ANgtADBJV2ZIALC3AMDOEAuyAAgMADBRiIUpAAR7AGDIIyN4AISZABRG8lc88SuuEOcqAAB4mbI8uSQ5RYFbCC1xB1dXLh4ozkkXKxQ2YQJhmkAuwnmZGTKBNA/g88wAAKCRFRHgg/P9eM4Ors7ONo62Dl8t6r8G/yJiYuP+5c+rcEAAAOF0ftH+LC+zGoA7BoBt/qIl7gRoXgugdfeLZrIPQLUAoOnaV/Nw+H48PEWhkLnZ2eXk5NhKxEJbYcpXff5nwl/AV/1s+X48/Pf14L7iJIEyXYFHBPjgwsz0TKUcz5IJhGLc5o9H/LcL//wd0yLESWK5WCoU41EScY5EmozzMqUiiUKSKcUl0v9k4t8s+wM+3zUAsGo+AXuRLahdYwP2SycQWHTA4vcAAPK7b8HUKAgDgGiD4c93/+8//UegJQCAZkmScQAAXkQkLlTKsz/HCAAARKCBKrBBG/TBGCzABhzBBdzBC/xgNoRCJMTCQhBCCmSAHHJgKayCQiiGzbAdKmAv1EAdNMBRaIaTcA4uwlW4Dj1wD/phCJ7BKLyBCQRByAgTYSHaiAFiilgjjggXmYX4IcFIBBKLJCDJiBRRIkuRNUgxUopUIFVIHfI9cgI5h1xGupE7yAAygvyGvEcxlIGyUT3UDLVDuag3GoRGogvQZHQxmo8WoJvQcrQaPYw2oefQq2gP2o8+Q8cwwOgYBzPEbDAuxsNCsTgsCZNjy7EirAyrxhqwVqwDu4n1Y8+xdwQSgUXACTYEd0IgYR5BSFhMWE7YSKggHCQ0EdoJNwkDhFHCJyKTqEu0JroR+cQYYjIxh1hILCPWEo8TLxB7iEPENyQSiUMyJ7mQAkmxpFTSEtJG0m5SI+ksqZs0="


def check_service_availability():
    """检查Ollama服务是否正常运行"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=5)
        return response.status_code == 200
    except (Timeout, ConnectionError, HTTPError) as e:
        print(f"❌ 服务不可用：{e}")
        return False


def get_local_models():
    """读取本地完整模型列表（保留模型名:标签）"""
    if not check_service_availability():
        return []
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=5)
        return [model["name"] for model in response.json().get("models", []) if model.get("name")]
    except Exception as e:
        print(f"❌ 读取模型列表失败：{e}")
        return []


def is_embedding_model(model_name):
    """纯能力验证：是否为Embedding模型（专属/api/embeddings接口）"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/embeddings",
            json={"model": model_name, "prompt": "test"},
            timeout=10
        )
        if response.status_code != 200:
            return False
        res_json = response.json()
        return "embedding" in res_json and isinstance(res_json["embedding"], list) and len(res_json["embedding"]) > 128
    except Exception:
        return False


def is_vision_model(model_name):
    """纯能力验证：是否为Vision模型（支持图片输入）"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/chat",
            json={
                "model": model_name,
                "stream": False,
                "messages": [{"role": "user", "content": "描述图片", "images": [VALID_SIMPLE_IMAGE_BASE64]}]
            },
            timeout=15
        )
        if response.status_code != 200:
            return False
        res_json = response.json()
        content = res_json.get("message", {}).get("content", "").lower()
        # 过滤"不支持图片"类响应（基于响应内容，非模型名称关键词）
        error_phrases = ["不支持", "无法处理", "invalid", "no support", "cannot process"]
        return not any(phrase in content for phrase in error_phrases) and content.strip() != ""
    except Exception:
        return False


def classify_model(model_name):
    """无关键词分类：仅基于能力验证"""
    if is_embedding_model(model_name):
        return "Embedding模型"
    elif is_vision_model(model_name):
        return "Vision（图文）模型"
    else:
        # 验证纯文本LLM能力
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/generate",
                json={"model": model_name, "prompt": "你好"},
                timeout=10
            )
            if response.status_code == 200 and response.json().get("response", "").strip():
                return "纯文本LLM"
            return "未知模型（无匹配能力）"
        except Exception:
            return "未知模型（调用失败）"


def main():
    print("=" * 60)
    print("🔍 读取本地Ollama模型并分类（无任何关键词匹配）")
    print("=" * 60)

    local_models = get_local_models()
    if not local_models:
        print("❌ 未获取到本地模型")
        return

    print(f"✅ 共读取 {len(local_models)} 个模型\n")
    print("📊 分类结果：")
    print("-" * 60)
    for model in local_models:
        model_type = classify_model(model)
        print(f"模型名称：{model}")
        print(f"模型类型：{model_type}")
        print("-" * 60)


if __name__ == "__main__":
    main()