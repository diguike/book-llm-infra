"""
第 5 章 示例 1：vLLM 离线推理快速上手

演示 vLLM 的离线（offline）推理模式，直接在 Python 中调用。
适合批量推理、测试、脚本场景。
"""

from vllm import LLM, SamplingParams


def basic_generation():
    """基础文本生成"""
    print("=" * 60)
    print("1. 基础文本生成")
    print("=" * 60)

    # 初始化 vLLM 引擎
    # 首次运行会下载模型，约 1GB（0.5B 模型）
    llm = LLM(
        model="Qwen/Qwen2-0.5B-Instruct",
        dtype="auto",               # 自动选择精度
        max_model_len=2048,          # 最大序列长度
        gpu_memory_utilization=0.8,  # GPU 显存利用率
    )

    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        stop=["<|im_end|>"],
    )

    # 批量推理 —— vLLM 自动做 Continuous Batching
    prompts = [
        "<|im_start|>user\nWhat is Python?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain HTTP in one sentence.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt[:50] + "..."
        generated = output.outputs[0].text.strip()
        print(f"\n输入: {prompt}")
        print(f"输出: {generated}")
        print(f"Token 数: {len(output.outputs[0].token_ids)}")

    return llm


def chat_api(llm: LLM):
    """使用 vLLM 的 chat API（自动处理 chat template）"""
    print("\n" + "=" * 60)
    print("2. Chat API（自动套用 chat template）")
    print("=" * 60)

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy
        max_tokens=150,
    )

    # 使用 chat 接口，自动处理 chat template
    messages_list = [
        [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function to reverse a string."},
        ],
        [
            {"role": "user", "content": "What are the main HTTP methods?"},
        ],
    ]

    outputs = llm.chat(messages_list, sampling_params)

    for i, output in enumerate(outputs):
        print(f"\n--- 对话 {i + 1} ---")
        generated = output.outputs[0].text.strip()
        print(f"输出: {generated[:300]}")


def benchmark_throughput(llm: LLM):
    """简单的吞吐量测试"""
    import time

    print("\n" + "=" * 60)
    print("3. 吞吐量测试")
    print("=" * 60)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
    )

    # 构造 100 个请求
    num_requests = 100
    prompts = [
        f"<|im_start|>user\nCount from 1 to 10. Request #{i}<|im_end|>\n<|im_start|>assistant\n"
        for i in range(num_requests)
    ]

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    total_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    total_input_tokens = sum(
        len(output.prompt_token_ids) for output in outputs
    )

    print(f"请求数:          {num_requests}")
    print(f"总输入 tokens:    {total_input_tokens}")
    print(f"总输出 tokens:    {total_output_tokens}")
    print(f"总耗时:          {elapsed:.2f}s")
    print(f"吞吐量:          {num_requests / elapsed:.1f} requests/s")
    print(f"输出 token 吞吐:  {total_output_tokens / elapsed:.0f} tokens/s")


def main():
    llm = basic_generation()
    chat_api(llm)
    benchmark_throughput(llm)


if __name__ == "__main__":
    main()
