"""
第 6 章 示例 1：Ollama 本地推理

演示如何使用 Ollama 进行本地 LLM 推理。
支持两种调用方式：
1. Ollama 原生 API (http://localhost:11434)
2. OpenAI 兼容 API (http://localhost:11434/v1)

前提：安装并启动 Ollama
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen2:0.5b
"""

import json
import time
import requests
from typing import Generator, Optional


OLLAMA_BASE_URL = "http://localhost:11434"


# ============================================================
# 1. Ollama 原生 API
# ============================================================

def ollama_generate(prompt: str, model: str = "qwen2:0.5b") -> str:
    """Ollama 原生 generate API（非对话模式）"""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 100,
            },
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["response"]


def ollama_chat(messages: list[dict], model: str = "qwen2:0.5b") -> str:
    """Ollama 原生 chat API"""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 200,
            },
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def ollama_chat_stream(
    messages: list[dict],
    model: str = "qwen2:0.5b",
) -> Generator[str, None, None]:
    """Ollama 原生 Streaming chat API"""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_predict": 200,
            },
        },
        stream=True,
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if not data.get("done", False):
                yield data["message"]["content"]


# ============================================================
# 2. OpenAI 兼容 API（通过 OpenAI SDK）
# ============================================================

def ollama_openai_compat(model: str = "qwen2:0.5b"):
    """用 OpenAI SDK 调用 Ollama"""
    from openai import OpenAI

    # Ollama 在 /v1 路径提供 OpenAI 兼容接口
    client = OpenAI(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",  # Ollama 不验证 key，但 SDK 要求非空
    )

    # 非 streaming
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is Docker?"},
        ],
        temperature=0.3,
        max_tokens=100,
    )
    return response.choices[0].message.content


# ============================================================
# 3. 模型管理
# ============================================================

def list_models() -> list[dict]:
    """列出本地已下载的模型"""
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
    response.raise_for_status()
    return response.json().get("models", [])


def pull_model(model: str):
    """下载模型（streaming 显示进度）"""
    print(f"拉取模型: {model}")
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/pull",
        json={"model": model},
        stream=True,
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            status = data.get("status", "")
            completed = data.get("completed", 0)
            total = data.get("total", 0)
            if total > 0:
                pct = completed / total * 100
                print(f"\r  {status}: {pct:.1f}%", end="", flush=True)
            else:
                print(f"\r  {status}", end="", flush=True)
    print()


def model_info(model: str) -> dict:
    """获取模型详细信息"""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/show",
        json={"model": model},
    )
    response.raise_for_status()
    return response.json()


# ============================================================
# 主程序
# ============================================================

def main():
    model = "qwen2:0.5b"

    print("=" * 60)
    print("Ollama 本地推理示例")
    print("=" * 60)

    # 检查 Ollama 是否运行
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
    except requests.ConnectionError:
        print(f"无法连接 Ollama ({OLLAMA_BASE_URL})")
        print("请先安装并启动 Ollama:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        print("  ollama serve")
        return

    # 1. 列出本地模型
    print("\n--- 本地模型 ---")
    models = list_models()
    if models:
        for m in models:
            size_gb = m.get("size", 0) / 1e9
            print(f"  {m['name']} ({size_gb:.1f} GB)")
    else:
        print("  无本地模型，正在下载...")
        pull_model(model)

    # 确保模型存在
    model_names = [m["name"] for m in list_models()]
    if not any(model in name for name in model_names):
        pull_model(model)

    # 2. 基础生成
    print(f"\n--- 基础生成 (generate API) ---")
    start = time.perf_counter()
    result = ollama_generate("What is Python? Answer in one sentence.", model)
    elapsed = time.perf_counter() - start
    print(f"  回复: {result.strip()}")
    print(f"  耗时: {elapsed:.2f}s")

    # 3. Chat API
    print(f"\n--- Chat API ---")
    result = ollama_chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain REST API in 2 sentences."},
    ], model)
    print(f"  回复: {result.strip()}")

    # 4. Streaming
    print(f"\n--- Streaming Chat ---")
    print("  回复: ", end="", flush=True)
    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    for token in ollama_chat_stream([
        {"role": "user", "content": "Count from 1 to 10."},
    ], model):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        print(token, end="", flush=True)
        token_count += 1

    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) * 1000 if first_token_time else 0
    print(f"\n  TTFT: {ttft:.0f}ms, 总耗时: {elapsed:.2f}s")

    # 5. OpenAI 兼容调用
    print(f"\n--- OpenAI SDK 兼容调用 ---")
    try:
        result = ollama_openai_compat(model)
        print(f"  回复: {result.strip()}")
    except ImportError:
        print("  需要安装 openai: pip install openai")
    except Exception as e:
        print(f"  错误: {e}")

    # 6. 模型信息
    print(f"\n--- 模型信息 ---")
    try:
        info = model_info(model)
        details = info.get("details", {})
        print(f"  Family:     {details.get('family', 'N/A')}")
        print(f"  Parameters: {details.get('parameter_size', 'N/A')}")
        print(f"  Quant:      {details.get('quantization_level', 'N/A')}")
        print(f"  Format:     {details.get('format', 'N/A')}")
    except Exception as e:
        print(f"  获取失败: {e}")


if __name__ == "__main__":
    main()
