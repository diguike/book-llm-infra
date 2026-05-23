"""
第 5 章 示例 3：对比 HuggingFace generate 和 vLLM 的吞吐量

在相同模型和硬件上，对比两种推理方式的性能差异。
预期结果：vLLM 在批量场景下吞吐量高 5-15 倍。
"""

import time
import torch
import json
from typing import Optional


def benchmark_hf_generate(
    model_name: str,
    prompts: list[str],
    max_new_tokens: int = 100,
    batch_size: int = 1,
) -> dict:
    """使用 HuggingFace generate 进行推理"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[HF generate] 加载模型 (batch_size={batch_size})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    # 预热
    warmup_input = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(**warmup_input, max_new_tokens=5)

    total_output_tokens = 0
    start = time.perf_counter()

    # 按 batch 处理
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 计算生成的 token 数
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            total_output_tokens += len(output) - input_len

    elapsed = time.perf_counter() - start

    # 清理
    del model
    torch.cuda.empty_cache()

    result = {
        "method": f"HF generate (bs={batch_size})",
        "num_requests": len(prompts),
        "total_output_tokens": total_output_tokens,
        "total_time_s": elapsed,
        "throughput_req_s": len(prompts) / elapsed,
        "throughput_tok_s": total_output_tokens / elapsed,
    }

    print(f"[HF generate] 完成: {result['throughput_req_s']:.1f} req/s, "
          f"{result['throughput_tok_s']:.0f} tok/s")
    return result


def benchmark_vllm(
    model_name: str,
    prompts: list[str],
    max_new_tokens: int = 100,
) -> dict:
    """使用 vLLM 进行推理"""
    from vllm import LLM, SamplingParams

    print(f"\n[vLLM] 加载模型...")
    llm = LLM(
        model=model_name,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        temperature=0,  # greedy，与 HF 保持一致
        max_tokens=max_new_tokens,
    )

    # 预热
    _ = llm.generate(["Hello"], sampling_params)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    total_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )

    # 清理
    del llm
    torch.cuda.empty_cache()

    result = {
        "method": "vLLM",
        "num_requests": len(prompts),
        "total_output_tokens": total_output_tokens,
        "total_time_s": elapsed,
        "throughput_req_s": len(prompts) / elapsed,
        "throughput_tok_s": total_output_tokens / elapsed,
    }

    print(f"[vLLM] 完成: {result['throughput_req_s']:.1f} req/s, "
          f"{result['throughput_tok_s']:.0f} tok/s")
    return result


def print_comparison(results: list[dict]):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("性能对比结果")
    print("=" * 80)
    print(f"{'方案':<30} {'请求数':<8} {'总 token':<10} "
          f"{'耗时 (s)':<10} {'req/s':<10} {'tok/s':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['method']:<30} {r['num_requests']:<8} "
              f"{r['total_output_tokens']:<10} "
              f"{r['total_time_s']:<10.2f} "
              f"{r['throughput_req_s']:<10.1f} "
              f"{r['throughput_tok_s']:<10.0f}")

    # ASCII 柱状图
    if len(results) > 1:
        max_tok_s = max(r["throughput_tok_s"] for r in results)
        print(f"\n吞吐量对比 (tokens/s):")
        for r in results:
            bar_len = int(r["throughput_tok_s"] / max_tok_s * 40)
            bar = "█" * bar_len
            print(f"  {r['method']:<30} | {bar} {r['throughput_tok_s']:.0f}")

        # 计算加速比
        baseline = results[0]["throughput_tok_s"]
        print(f"\n加速比（相对于 {results[0]['method']}）:")
        for r in results[1:]:
            speedup = r["throughput_tok_s"] / baseline
            print(f"  {r['method']}: {speedup:.1f}x")


def main():
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    max_new_tokens = 100

    # 构造测试 prompts
    test_prompts = [
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        for q in [
            "What is machine learning?",
            "Explain HTTP status codes.",
            "What is Docker?",
            "How does DNS work?",
            "What is REST API?",
            "Explain TCP vs UDP.",
            "What is a database index?",
            "How does HTTPS work?",
            "What is microservices architecture?",
            "Explain OAuth 2.0.",
        ] * 5  # 50 个请求
    ]

    num_requests = len(test_prompts)
    print(f"模型:     {model_name}")
    print(f"请求数:   {num_requests}")
    print(f"输出长度: {max_new_tokens} tokens/request")

    if not torch.cuda.is_available():
        print("\n⚠ 未检测到 GPU，此 benchmark 需要 CUDA GPU")
        return

    print(f"GPU:      {torch.cuda.get_device_name()}")
    print(f"显存:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    results = []

    # 1. HF generate (batch=1)
    results.append(benchmark_hf_generate(
        model_name, test_prompts, max_new_tokens, batch_size=1
    ))

    # 2. HF generate (batch=8)
    results.append(benchmark_hf_generate(
        model_name, test_prompts, max_new_tokens, batch_size=8
    ))

    # 3. vLLM
    results.append(benchmark_vllm(
        model_name, test_prompts, max_new_tokens
    ))

    print_comparison(results)


if __name__ == "__main__":
    main()
