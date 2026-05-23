"""Speculative Decoding 概念演示

核心思想: 小模型快速猜测 K 个 token，大模型一次性并行验证。
由于大模型 decode 是 memory-bound (GPU 算力闲置)，验证 K 个 token
和生成 1 个 token 的耗时相近，因此每步可以获得多个 token。

使用方法:
    # 概念模拟 (不需要 GPU/模型)
    python 02_speculative_decoding.py

    # 调整参数
    python 02_speculative_decoding.py --acceptance-rate 0.8 --k 7 --tokens 100

    # 用 vLLM 实际测试 (需要 GPU + vLLM)
    python 02_speculative_decoding.py --vllm --model Qwen/Qwen2-7B --draft-model Qwen/Qwen2-0.5B
"""

import argparse
import random
import time


def simulate_speculative_decoding(n_tokens=50, k=5, acceptance_rate=0.7):
    """模拟 speculative decoding 的过程和加速效果"""

    # 时间参数 (毫秒), 基于 7B + 0.5B 模型的典型值
    DRAFT_MS = 5        # 小模型每 token 耗时
    TARGET_DECODE_MS = 30   # 大模型标准 decode 每 token 耗时
    TARGET_VERIFY_MS = 35   # 大模型验证 K 个 token 耗时 (并行, 接近单 token)

    random.seed(42)

    print(f"参数: 生成 {n_tokens} tokens, K={k}, 接受率={acceptance_rate:.0%}")
    print(f"Draft model: {DRAFT_MS}ms/token, Target model: {TARGET_DECODE_MS}ms/token")
    print()

    # === 标准自回归 ===
    standard_time = n_tokens * TARGET_DECODE_MS

    # === Speculative Decoding ===
    generated = 0
    spec_time = 0
    steps = 0
    total_accepted = 0
    total_drafted = 0

    print("Speculative Decoding 过程:")
    while generated < n_tokens:
        steps += 1

        # Draft model 生成 K 个候选
        draft_time = DRAFT_MS * k
        spec_time += draft_time

        # Target model 验证
        spec_time += TARGET_VERIFY_MS

        # 模拟逐个接受/拒绝
        accepted = 0
        for i in range(k):
            if random.random() < acceptance_rate:
                accepted += 1
            else:
                break  # 第一个拒绝位置之后全部丢弃

        # 即使全部拒绝, target model 也会在拒绝位置生成 1 个正确 token
        tokens_this_step = accepted + 1
        generated += tokens_this_step
        total_accepted += accepted
        total_drafted += k

        if steps <= 5 or generated >= n_tokens:
            status = "✓" * accepted + "✗" + "·" * (k - accepted - 1) if accepted < k else "✓" * k
            print(f"  Step {steps:>3}: 猜 {k} 个 [{status}] → 得 {tokens_this_step} 个 "
                  f"(累计 {min(generated, n_tokens)}/{n_tokens})")
        elif steps == 6:
            print(f"  ...")

    actual_rate = total_accepted / total_drafted if total_drafted > 0 else 0

    print()
    print("=" * 55)
    print(f"{'指标':<25} {'标准自回归':<15} {'Speculative':<15}")
    print("-" * 55)
    print(f"{'总步数':<25} {n_tokens:<15} {steps:<15}")
    print(f"{'总耗时':<25} {standard_time:<15.0f} {spec_time:<15.0f}")
    print(f"{'速度 (tokens/s)':<25} {n_tokens/standard_time*1000:<15.1f} {generated/spec_time*1000:<15.1f}")
    print(f"{'加速比':<25} {'1.00x':<15} {standard_time/spec_time:.2f}x")
    print(f"{'实际接受率':<25} {'-':<15} {actual_rate:.1%}")


def sweep_acceptance_rates(k=5):
    """展示不同接受率下的加速效果"""
    DRAFT_MS = 5
    TARGET_DECODE_MS = 30
    TARGET_VERIFY_MS = 35

    print()
    print("=" * 55)
    print(f"接受率 vs 加速比 (K={k})")
    print("=" * 55)

    for rate in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        # 计算期望的每步 token 数
        # 接受序列长度的期望值: sum(rate^i for i in 1..k)
        expected_accepted = sum(rate ** i for i in range(1, k + 1))
        tokens_per_step = expected_accepted + 1  # +1 for target model's correction

        step_cost = DRAFT_MS * k + TARGET_VERIFY_MS
        spec_tps = tokens_per_step / step_cost * 1000
        standard_tps = 1000 / TARGET_DECODE_MS
        speedup = spec_tps / standard_tps

        bar_len = int(speedup * 12)
        bar = "█" * bar_len
        print(f"  {rate:>4.0%}  {speedup:.2f}x  {bar}")

    print()
    print("  接受率取决于 draft model 和 target model 的匹配度。")
    print("  同系列模型 (如 Qwen2-0.5B → Qwen2-7B) 通常有 70-85% 接受率。")


def vllm_benchmark(model, draft_model):
    """用 vLLM 实际对比 speculative decoding"""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("需要安装 vLLM: pip install vllm")
        return

    prompts = [
        "请详细解释 Transformer 架构中 Self-Attention 的工作原理，",
        "Python 异步编程的核心概念和最佳实践是什么？",
        "如何设计一个高可用的微服务架构？",
    ]
    params = SamplingParams(temperature=0, max_tokens=200)

    # 标准推理
    print(f"[1/2] 标准推理 ({model}) ...")
    llm_standard = LLM(model=model)

    start = time.perf_counter()
    outputs_standard = llm_standard.generate(prompts, params)
    time_standard = time.perf_counter() - start

    total_tokens_std = sum(len(o.outputs[0].token_ids) for o in outputs_standard)
    del llm_standard
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()

    # Speculative Decoding
    print(f"[2/2] Speculative Decoding ({model} + {draft_model}) ...")
    llm_spec = LLM(
        model=model,
        speculative_model=draft_model,
        num_speculative_tokens=5,
    )

    start = time.perf_counter()
    outputs_spec = llm_spec.generate(prompts, params)
    time_spec = time.perf_counter() - start

    total_tokens_spec = sum(len(o.outputs[0].token_ids) for o in outputs_spec)

    print()
    print("=" * 50)
    print(f"{'指标':<20} {'标准':<15} {'Speculative':<15}")
    print("-" * 50)
    print(f"{'总耗时 (s)':<20} {time_standard:<15.2f} {time_spec:<15.2f}")
    print(f"{'总 tokens':<20} {total_tokens_std:<15} {total_tokens_spec:<15}")
    print(f"{'吞吐 (tok/s)':<20} {total_tokens_std/time_standard:<15.1f} {total_tokens_spec/time_spec:<15.1f}")
    print(f"{'加速比':<20} {'1.00x':<15} {time_standard/time_spec:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding 演示")
    parser.add_argument("--tokens", type=int, default=50, help="生成的 token 数")
    parser.add_argument("--k", type=int, default=5, help="每步猜测的 token 数")
    parser.add_argument("--acceptance-rate", type=float, default=0.7, help="模拟接受率")
    parser.add_argument("--vllm", action="store_true", help="用 vLLM 实际运行")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B")
    parser.add_argument("--draft-model", type=str, default="Qwen/Qwen2-0.5B")
    args = parser.parse_args()

    if args.vllm:
        vllm_benchmark(args.model, args.draft_model)
    else:
        simulate_speculative_decoding(args.tokens, args.k, args.acceptance_rate)
        sweep_acceptance_rates(args.k)


if __name__ == "__main__":
    main()
