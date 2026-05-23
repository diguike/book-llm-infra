"""Prefix Caching 效果演示

当多个请求共享相同的 system prompt (如 Agent 的 tool description)，
prefix caching 可以跳过 prefix 部分的 Prefill 计算，大幅降低 TTFT。

使用方法:
    # 模拟演示 (不需要 GPU)
    python 03_prefix_caching_demo.py

    # vLLM 实际测试
    python 03_prefix_caching_demo.py --vllm --model Qwen/Qwen2-7B
"""

import argparse
import time


# 模拟典型 Agent system prompt (包含 tool definitions)
SYSTEM_PROMPT = """You are a helpful assistant with access to the following tools:

1. search_web(query: str) -> str
   Search the web for information. Returns relevant text snippets from top results.
   Use this when the user asks about current events, facts you're unsure about,
   or anything that requires up-to-date information.

2. read_file(path: str) -> str
   Read the contents of a file at the given path. Returns the full text content.
   Supports text files, JSON, YAML, Markdown, and source code files.

3. write_file(path: str, content: str) -> bool
   Write content to a file at the given path. Creates the file if it doesn't exist,
   overwrites if it does. Returns True on success, False on failure.

4. run_code(language: str, code: str) -> str
   Execute code in a sandboxed environment. Supported languages: python, javascript,
   bash. Returns stdout output. Timeout: 30 seconds. No network access.

5. query_database(sql: str) -> list[dict]
   Execute a read-only SQL query against the connected PostgreSQL database.
   Returns results as a list of dictionaries. Max 1000 rows.

6. send_email(to: str, subject: str, body: str) -> bool
   Send an email to the specified address. Body supports plain text and HTML.

7. create_calendar_event(title: str, start: str, end: str, attendees: list[str]) -> dict
   Create a calendar event. Times in ISO 8601 format. Returns event details.

8. get_weather(city: str, days: int = 1) -> dict
   Get weather forecast. Returns temperature, humidity, conditions, and forecast.

When using tools, respond with a JSON object: {"tool": "<name>", "arguments": {...}}
Always explain your reasoning before calling a tool. If a tool call fails, try an
alternative approach. Never make up information - use search_web if unsure.
"""

USER_QUERIES = [
    "帮我搜索一下最近的 vLLM 版本更新",
    "读取 /home/user/config.yaml 文件内容",
    "今天北京天气怎么样",
    "帮我写一个 Python 快排实现并运行",
    "查询数据库中最近 10 条订单",
    "发邮件给 team@example.com 通知明天的会议取消",
    "创建一个明天下午 2 点到 3 点的团队周会",
    "帮我搜索 FlashAttention 3 的最新论文",
    "读取 /var/log/app.log 最近的错误",
    "查一下上海未来三天的天气预报",
]


def simulate_prefix_caching():
    """模拟 prefix caching 的效果"""

    # 粗略估计 token 数
    system_tokens = len(SYSTEM_PROMPT.split())  # ~250 tokens (英文)
    prefill_ms_per_token = 0.08  # A100 上大约 0.05-0.1 ms/token

    print(f"System prompt: ~{system_tokens} tokens")
    print(f"用户请求数量: {len(USER_QUERIES)}")
    print(f"Prefill 速度: ~{prefill_ms_per_token} ms/token (A100, 7B 模型)")
    print()

    # ===== 无 Prefix Caching =====
    print("━" * 60)
    print("【无 Prefix Caching】 每次请求完整 Prefill")
    print("━" * 60)

    total_no_cache = 0
    for i, query in enumerate(USER_QUERIES):
        query_tokens = max(len(query) // 2, 5)  # 中文粗略估计
        total_tokens = system_tokens + query_tokens
        prefill_ms = total_tokens * prefill_ms_per_token

        total_no_cache += prefill_ms
        if i < 3 or i == len(USER_QUERIES) - 1:
            print(f"  请求 {i+1:>2}: prefill {total_tokens:>4} tokens = {prefill_ms:>6.1f} ms")
        elif i == 3:
            print(f"  ...")

    print(f"  {'总 Prefill 时间':>32}: {total_no_cache:.1f} ms")

    print()

    # ===== 有 Prefix Caching =====
    print("━" * 60)
    print("【有 Prefix Caching】 首次缓存，后续复用")
    print("━" * 60)

    total_with_cache = 0
    for i, query in enumerate(USER_QUERIES):
        query_tokens = max(len(query) // 2, 5)

        if i == 0:
            # 第一次请求: 完整 prefill + 建立缓存
            total_tokens = system_tokens + query_tokens
            prefill_ms = total_tokens * prefill_ms_per_token
            print(f"  请求 {i+1:>2}: prefill {total_tokens:>4} tokens = {prefill_ms:>6.1f} ms "
                  f"(cache MISS → 缓存 {system_tokens} tokens)")
        else:
            # 后续请求: 只 prefill 用户 query 部分
            prefill_ms = query_tokens * prefill_ms_per_token
            if i < 3 or i == len(USER_QUERIES) - 1:
                print(f"  请求 {i+1:>2}: prefill {query_tokens:>4} tokens = {prefill_ms:>6.1f} ms "
                      f"(cache HIT → 跳过 {system_tokens} tokens)")
            elif i == 3:
                print(f"  ...")

        total_with_cache += prefill_ms

    print(f"  {'总 Prefill 时间':>32}: {total_with_cache:.1f} ms")

    # 对比
    print()
    print("━" * 60)
    print("对比")
    print("━" * 60)
    saving_pct = (1 - total_with_cache / total_no_cache) * 100
    avg_no_cache = total_no_cache / len(USER_QUERIES)
    avg_with_cache = total_with_cache / len(USER_QUERIES)

    print(f"  总 Prefill 节省: {saving_pct:.0f}%")
    print(f"  平均 TTFT (无缓存): {avg_no_cache:.1f} ms")
    print(f"  平均 TTFT (有缓存): {avg_with_cache:.1f} ms")
    print(f"  TTFT 降低: {(1 - avg_with_cache / avg_no_cache) * 100:.0f}%")

    print()
    print("实际价值:")
    print(f"  - System prompt 越长 (tool 越多), 节省越多")
    print(f"  - 请求量越大, 首次 miss 的成本被摊薄")
    print(f"  - vLLM 启用: --enable-prefix-caching")
    print(f"  - 云 API: Anthropic 缓存命中只收 10% 费用")


def vllm_prefix_caching(model):
    """用 vLLM 实际测试 prefix caching"""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("需要安装 vLLM: pip install vllm")
        return

    params = SamplingParams(temperature=0.7, max_tokens=100)
    prompts = [f"{SYSTEM_PROMPT}\n\nUser: {q}\nAssistant:" for q in USER_QUERIES]

    # 无 prefix caching
    print(f"[1/2] 无 Prefix Caching ({model}) ...")
    llm = LLM(model=model, enable_prefix_caching=False)

    start = time.perf_counter()
    llm.generate(prompts, params)
    time_no_cache = time.perf_counter() - start
    del llm

    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    # 有 prefix caching
    print(f"[2/2] 有 Prefix Caching ({model}) ...")
    llm = LLM(model=model, enable_prefix_caching=True)

    # 预热一次建立缓存
    llm.generate([prompts[0]], params)

    start = time.perf_counter()
    llm.generate(prompts, params)
    time_with_cache = time.perf_counter() - start

    print()
    print(f"结果:")
    print(f"  无 Prefix Caching: {time_no_cache:.2f}s")
    print(f"  有 Prefix Caching: {time_with_cache:.2f}s")
    print(f"  加速比: {time_no_cache / time_with_cache:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Prefix Caching 演示")
    parser.add_argument("--vllm", action="store_true", help="用 vLLM 实际测试")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B")
    args = parser.parse_args()

    if args.vllm:
        vllm_prefix_caching(args.model)
    else:
        simulate_prefix_caching()


if __name__ == "__main__":
    main()
