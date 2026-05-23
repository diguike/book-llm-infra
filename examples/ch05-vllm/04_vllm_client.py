"""
第 5 章 示例 4：用 OpenAI SDK 调用 vLLM

演示 vLLM 的 OpenAI 兼容 API，包括：
- 基础对话
- Streaming
- 多轮对话
- 工具调用 (Function Calling)
- JSON 模式

前提：先启动 vLLM server（见 02_vllm_server.sh）
"""

import json
import time
import asyncio
from typing import Optional

from openai import OpenAI, AsyncOpenAI


# ============================================================
# 配置
# ============================================================

VLLM_BASE_URL = "http://localhost:8000/v1"
API_KEY = "not-needed"  # vLLM 默认不需要 key
MODEL = "qwen2-7b"      # served-model-name


def get_client() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)


def get_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)


# ============================================================
# 1. 基础对话
# ============================================================

def basic_chat():
    """最简单的对话调用"""
    print("=" * 60)
    print("1. 基础对话")
    print("=" * 60)

    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is vLLM?"},
        ],
        temperature=0.7,
        max_tokens=200,
    )

    print(f"回复: {response.choices[0].message.content}")
    print(f"Token 用量: prompt={response.usage.prompt_tokens}, "
          f"completion={response.usage.completion_tokens}")


# ============================================================
# 2. Streaming 对话
# ============================================================

def streaming_chat():
    """Streaming 输出"""
    print("\n" + "=" * 60)
    print("2. Streaming 对话")
    print("=" * 60)

    client = get_client()

    start = time.perf_counter()
    first_token_time = None

    print("回复: ", end="", flush=True)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Write a haiku about programming."},
        ],
        temperature=0.7,
        max_tokens=100,
        stream=True,
    )

    token_count = 0
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            print(content, end="", flush=True)
            token_count += 1

    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) * 1000 if first_token_time else 0

    print(f"\n\nTTFT: {ttft:.0f}ms")
    print(f"总耗时: {elapsed * 1000:.0f}ms")
    print(f"生成 token 数: {token_count}")
    if token_count > 1 and first_token_time:
        decode_time = elapsed - (first_token_time - start)
        print(f"TPS: {(token_count - 1) / decode_time:.1f}")


# ============================================================
# 3. 多轮对话
# ============================================================

def multi_turn_chat():
    """多轮对话示例"""
    print("\n" + "=" * 60)
    print("3. 多轮对话")
    print("=" * 60)

    client = get_client()
    messages = [
        {"role": "system", "content": "You are a helpful math tutor. Be concise."},
    ]

    questions = [
        "What is a derivative?",
        "Can you give me a simple example?",
        "What about the derivative of x^3?",
    ]

    for q in questions:
        messages.append({"role": "user", "content": q})
        print(f"\nUser: {q}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=150,
        )

        answer = response.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})
        print(f"Assistant: {answer}")


# ============================================================
# 4. JSON 模式
# ============================================================

def json_mode():
    """JSON 结构化输出"""
    print("\n" + "=" * 60)
    print("4. JSON 模式")
    print("=" * 60)

    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "Extract information from the text and respond in JSON format "
                "with keys: name, age, occupation."
            )},
            {"role": "user", "content": (
                "John Smith is a 35-year-old software engineer "
                "who lives in San Francisco."
            )},
        ],
        temperature=0.0,
        max_tokens=100,
        response_format={"type": "json_object"},
    )

    result = response.choices[0].message.content
    print(f"原始响应: {result}")

    try:
        parsed = json.loads(result)
        print(f"解析结果: {json.dumps(parsed, indent=2)}")
    except json.JSONDecodeError:
        print("JSON 解析失败")


# ============================================================
# 5. 并发请求 (async)
# ============================================================

async def concurrent_requests():
    """异步并发请求，测试 vLLM 的并发处理能力"""
    print("\n" + "=" * 60)
    print("5. 并发请求测试")
    print("=" * 60)

    client = get_async_client()
    num_requests = 20

    questions = [
        f"What is the number {i} in binary?" for i in range(num_requests)
    ]

    start = time.perf_counter()

    # 同时发送所有请求
    tasks = [
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": q}],
            temperature=0.0,
            max_tokens=30,
        )
        for q in questions
    ]

    responses = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    total_tokens = sum(r.usage.completion_tokens for r in responses)

    print(f"并发请求数:    {num_requests}")
    print(f"总耗时:       {elapsed:.2f}s")
    print(f"吞吐量:       {num_requests / elapsed:.1f} req/s")
    print(f"总输出 token: {total_tokens}")
    print(f"Token 吞吐:   {total_tokens / elapsed:.0f} tok/s")

    # 打印几个结果
    for i in range(min(3, len(responses))):
        print(f"\n  Q: {questions[i]}")
        print(f"  A: {responses[i].choices[0].message.content.strip()}")


# ============================================================
# 6. 列出模型
# ============================================================

def list_models():
    """查看 vLLM 提供的模型列表"""
    print("\n" + "=" * 60)
    print("6. 模型列表")
    print("=" * 60)

    client = get_client()
    models = client.models.list()

    for model in models.data:
        print(f"  - {model.id} (owned by: {model.owned_by})")


def main():
    print("vLLM Client 测试")
    print(f"Server: {VLLM_BASE_URL}")
    print(f"Model:  {MODEL}")
    print()

    try:
        # 先检查服务是否可用
        client = get_client()
        client.models.list()
    except Exception as e:
        print(f"无法连接 vLLM server: {e}")
        print(f"\n请先启动 vLLM server:")
        print(f"  ./02_vllm_server.sh")
        print(f"  # 或")
        print(f"  vllm serve Qwen/Qwen2-7B-Instruct --served-model-name {MODEL}")
        return

    list_models()
    basic_chat()
    streaming_chat()
    multi_turn_chat()
    json_mode()
    asyncio.run(concurrent_requests())


if __name__ == "__main__":
    main()
