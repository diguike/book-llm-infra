"""
第 6 章 示例 2：SGLang 结构化生成示例

演示 SGLang 的核心能力：
1. 基础推理
2. 结构化输出（JSON Schema / 正则约束）
3. 多轮对话的 RadixAttention 复用

前提：
  pip install "sglang[all]"
  python -m sglang.launch_server --model-path Qwen/Qwen2-0.5B-Instruct --port 30000
"""

import json
import time
import requests
from typing import Optional


SGLANG_BASE_URL = "http://localhost:30000"


# ============================================================
# 1. 基础推理（OpenAI 兼容）
# ============================================================

def basic_inference():
    """SGLang 提供 OpenAI 兼容的 API"""
    print("=" * 60)
    print("1. 基础推理 (OpenAI 兼容 API)")
    print("=" * 60)

    from openai import OpenAI

    client = OpenAI(
        base_url=f"{SGLANG_BASE_URL}/v1",
        api_key="not-needed",
    )

    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is SGLang?"},
        ],
        temperature=0.3,
        max_tokens=200,
    )

    print(f"回复: {response.choices[0].message.content}")
    print(f"Tokens: prompt={response.usage.prompt_tokens}, "
          f"completion={response.usage.completion_tokens}")


# ============================================================
# 2. 结构化输出 —— JSON Schema 约束
# ============================================================

def structured_json_output():
    """使用 JSON Schema 约束输出格式"""
    print("\n" + "=" * 60)
    print("2. 结构化 JSON 输出")
    print("=" * 60)

    from openai import OpenAI

    client = OpenAI(
        base_url=f"{SGLANG_BASE_URL}/v1",
        api_key="not-needed",
    )

    # 定义 JSON Schema
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "occupation": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
        },
        "required": ["name", "age", "occupation", "skills"],
    }

    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": (
                "Extract structured information from text. "
                "Return valid JSON matching the schema."
            )},
            {"role": "user", "content": (
                "Alice is a 28-year-old data scientist who specializes in "
                "machine learning, Python, SQL, and data visualization."
            )},
        ],
        temperature=0.0,
        max_tokens=200,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "person_info",
                "schema": person_schema,
            },
        },
    )

    result = response.choices[0].message.content
    print(f"原始响应: {result}")

    try:
        parsed = json.loads(result)
        print(f"解析结果:\n{json.dumps(parsed, indent=2)}")
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")


# ============================================================
# 3. 正则表达式约束
# ============================================================

def regex_constrained_output():
    """使用正则表达式约束输出"""
    print("\n" + "=" * 60)
    print("3. 正则表达式约束输出")
    print("=" * 60)

    # SGLang 的 generate 接口支持 regex 约束
    response = requests.post(
        f"{SGLANG_BASE_URL}/generate",
        json={
            "text": "Generate a US phone number: ",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 20,
                "regex": r"\(\d{3}\) \d{3}-\d{4}",  # (xxx) xxx-xxxx 格式
            },
        },
    )

    if response.ok:
        data = response.json()
        print(f"生成的电话号码: {data.get('text', 'N/A')}")
    else:
        print(f"请求失败: {response.status_code}")

    # 生成 email 格式
    response = requests.post(
        f"{SGLANG_BASE_URL}/generate",
        json={
            "text": "Generate an email address: ",
            "sampling_params": {
                "temperature": 0.5,
                "max_new_tokens": 30,
                "regex": r"[a-z]+\.[a-z]+@[a-z]+\.com",
            },
        },
    )

    if response.ok:
        data = response.json()
        print(f"生成的邮箱: {data.get('text', 'N/A')}")


# ============================================================
# 4. Batch 推理 + 前缀复用
# ============================================================

def batch_with_shared_prefix():
    """
    演示共享前缀的批量推理。
    SGLang 的 RadixAttention 会自动复用相同前缀的 KV Cache。
    """
    print("\n" + "=" * 60)
    print("4. 批量推理 (共享前缀, RadixAttention 自动复用)")
    print("=" * 60)

    from openai import OpenAI

    client = OpenAI(
        base_url=f"{SGLANG_BASE_URL}/v1",
        api_key="not-needed",
    )

    # 共同的 system prompt（长前缀）
    system_prompt = """You are a code review assistant. You analyze code and provide:
1. A brief summary of what the code does
2. Potential bugs or issues
3. Suggestions for improvement
Be concise and focus on the most important points."""

    # 不同的用户请求，但共享相同的 system prompt
    code_snippets = [
        "def add(a, b): return a + b",
        "def divide(a, b): return a / b",
        "def sort(arr): return sorted(arr, reverse=True)",
        "def greet(name): print('Hello ' + name)",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    ]

    start = time.perf_counter()
    results = []

    for code in code_snippets:
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Review this code:\n```python\n{code}\n```"},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        results.append(response.choices[0].message.content)

    elapsed = time.perf_counter() - start

    for i, (code, review) in enumerate(zip(code_snippets, results)):
        print(f"\n  代码 {i + 1}: {code}")
        print(f"  评审: {review.strip()[:150]}")

    print(f"\n  总耗时: {elapsed:.2f}s ({len(code_snippets)} 个请求)")
    print(f"  平均:   {elapsed / len(code_snippets):.2f}s/request")
    print(f"  (RadixAttention 自动复用 system prompt 的 KV Cache)")


# ============================================================
# 5. Agent 工具调用场景
# ============================================================

def agent_tool_calling():
    """模拟 Agent 场景的工具调用"""
    print("\n" + "=" * 60)
    print("5. Agent 工具调用场景")
    print("=" * 60)

    from openai import OpenAI

    client = OpenAI(
        base_url=f"{SGLANG_BASE_URL}/v1",
        api_key="not-needed",
    )

    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["city"],
                },
            },
        },
    ]

    try:
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with access to tools."},
                {"role": "user", "content": "What's the weather like in Tokyo?"},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=200,
        )

        choice = response.choices[0]
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                print(f"  工具调用: {tc.function.name}")
                print(f"  参数:     {tc.function.arguments}")
        else:
            print(f"  回复: {choice.message.content}")
    except Exception as e:
        print(f"  工具调用失败（模型可能不支持）: {e}")


def main():
    print("SGLang 结构化输出示例")
    print(f"Server: {SGLANG_BASE_URL}")
    print()

    # 检查 SGLang 是否运行
    try:
        resp = requests.get(f"{SGLANG_BASE_URL}/health", timeout=2)
    except requests.ConnectionError:
        print(f"无法连接 SGLang server ({SGLANG_BASE_URL})")
        print("请先启动 SGLang:")
        print("  pip install 'sglang[all]'")
        print("  python -m sglang.launch_server \\")
        print("    --model-path Qwen/Qwen2-0.5B-Instruct \\")
        print("    --port 30000")
        return

    basic_inference()
    structured_json_output()
    regex_constrained_output()
    batch_with_shared_prefix()
    agent_tool_calling()


if __name__ == "__main__":
    main()
