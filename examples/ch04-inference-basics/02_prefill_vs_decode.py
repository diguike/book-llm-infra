"""
第 4 章 示例 2：测量和可视化 Prefill vs Decode 时间

演示 LLM 推理的两个阶段：
- Prefill: 并行处理所有输入 token (compute-bound)
- Decode: 逐个生成 token (memory-bound)
"""

import time
import torch
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class InferenceMetrics:
    """推理指标"""
    input_tokens: int
    output_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    ttft_ms: float  # Time To First Token
    tps: float      # Tokens Per Second (decode)
    total_time_ms: float


def measure_prefill_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> InferenceMetrics:
    """
    精确测量 Prefill 和 Decode 阶段的耗时。

    通过手动控制生成过程来分别计量两个阶段。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    # 预热 GPU
    with torch.no_grad():
        _ = model(input_ids)

    # === Prefill 阶段 ===
    # 一次性处理所有输入 token，生成第一个输出 token
    torch.cuda.synchronize() if device == "cuda" else None
    prefill_start = time.perf_counter()

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    torch.cuda.synchronize() if device == "cuda" else None
    prefill_end = time.perf_counter()
    prefill_time = prefill_end - prefill_start

    generated_tokens = [next_token.item()]

    # === Decode 阶段 ===
    # 逐个生成后续 token
    torch.cuda.synchronize() if device == "cuda" else None
    decode_start = time.perf_counter()

    current_token = next_token
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(
                current_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            current_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_tokens.append(current_token.item())

        # 遇到 EOS 提前停止
        if current_token.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize() if device == "cuda" else None
    decode_end = time.perf_counter()
    decode_time = decode_end - decode_start

    num_decode_tokens = len(generated_tokens) - 1  # 第一个 token 属于 prefill
    tps = num_decode_tokens / decode_time if decode_time > 0 else 0

    metrics = InferenceMetrics(
        input_tokens=input_len,
        output_tokens=len(generated_tokens),
        prefill_time_ms=prefill_time * 1000,
        decode_time_ms=decode_time * 1000,
        ttft_ms=prefill_time * 1000,
        tps=tps,
        total_time_ms=(prefill_time + decode_time) * 1000,
    )

    # 打印生成的文本
    all_ids = torch.cat([input_ids[0], torch.tensor(generated_tokens)], dim=0)
    generated_text = tokenizer.decode(all_ids, skip_special_tokens=True)
    print(f"Generated: {generated_text[:200]}...")

    return metrics


def run_scaling_experiment(
    model,
    tokenizer,
    device: str = "cuda",
):
    """
    测试不同输入长度下 Prefill 和 Decode 的耗时变化。

    预期结果：
    - Prefill 时间随输入长度线性增长（compute-bound）
    - Decode 每个 token 的时间基本不变（memory-bound，主要受模型大小限制）
    """
    # 用重复的句子构造不同长度的输入
    base_text = "The quick brown fox jumps over the lazy dog. "
    test_configs = [
        ("short", base_text, 20),
        ("medium", base_text * 10, 20),
        ("long", base_text * 50, 20),
        ("very_long", base_text * 100, 20),
    ]

    results = []
    print("\n" + "=" * 80)
    print("Prefill vs Decode 扩展实验")
    print("=" * 80)

    for name, prompt, max_tokens in test_configs:
        # 先 tokenize 看看实际长度
        tokens = tokenizer(prompt, return_tensors="pt")
        actual_len = tokens["input_ids"].shape[1]

        print(f"\n--- {name} (输入 {actual_len} tokens, 输出 {max_tokens} tokens) ---")
        metrics = measure_prefill_decode(
            model, tokenizer, prompt,
            max_new_tokens=max_tokens,
            device=device,
        )
        results.append((name, actual_len, metrics))
        print_metrics(metrics)

    # 打印对比表格
    print("\n" + "=" * 80)
    print("对比汇总")
    print("-" * 80)
    print(f"{'配置':<12} {'输入 tokens':<12} {'Prefill (ms)':<14} "
          f"{'Decode (ms)':<14} {'TTFT (ms)':<12} {'TPS':<8}")
    print("-" * 80)
    for name, input_len, m in results:
        print(f"{name:<12} {input_len:<12} {m.prefill_time_ms:<14.1f} "
              f"{m.decode_time_ms:<14.1f} {m.ttft_ms:<12.1f} {m.tps:<8.1f}")

    # ASCII 可视化
    print("\n\nPrefill 时间 vs 输入长度 (越长越慢):")
    max_prefill = max(m.prefill_time_ms for _, _, m in results)
    for name, input_len, m in results:
        bar_len = int(m.prefill_time_ms / max_prefill * 40)
        bar = "█" * bar_len
        print(f"  {name:<12} ({input_len:>5} tok) | {bar} {m.prefill_time_ms:.1f}ms")

    print("\nDecode 每 token 耗时 (应该相对稳定):")
    for name, _, m in results:
        per_token = m.decode_time_ms / max(m.output_tokens - 1, 1)
        bar_len = int(per_token * 2)
        bar = "█" * min(bar_len, 40)
        print(f"  {name:<12}           | {bar} {per_token:.1f}ms/token")


def print_metrics(m: InferenceMetrics):
    """格式化打印推理指标"""
    print(f"  输入 tokens:    {m.input_tokens}")
    print(f"  输出 tokens:    {m.output_tokens}")
    print(f"  Prefill 耗时:   {m.prefill_time_ms:.1f} ms")
    print(f"  Decode 耗时:    {m.decode_time_ms:.1f} ms")
    print(f"  TTFT:           {m.ttft_ms:.1f} ms")
    print(f"  TPS:            {m.tps:.1f} tokens/s")
    print(f"  总耗时:         {m.total_time_ms:.1f} ms")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 使用小模型以便快速运行
    model_name = "Qwen/Qwen2-0.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"模型: {model_name}")
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    # 单次测量
    print("\n" + "=" * 80)
    print("单次推理测量")
    print("=" * 80)
    prompt = "Explain the difference between TCP and UDP in simple terms:"
    metrics = measure_prefill_decode(
        model, tokenizer, prompt,
        max_new_tokens=50,
        device=device,
    )
    print_metrics(metrics)

    # 扩展实验
    run_scaling_experiment(model, tokenizer, device)


if __name__ == "__main__":
    main()
