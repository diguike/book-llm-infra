#!/usr/bin/env python3
"""
对比有无 KV Cache 时的推理速度差异。

演示 KV Cache 的工作原理：
- 不用 KV Cache: 每生成一个 token，重新计算所有 token 的 attention
- 用 KV Cache: 只计算新 token 的 Q/K/V，复用之前缓存的 K/V

用法:
    python 03_kv_cache_demo.py
    python 03_kv_cache_demo.py --seq-length 256 --gen-tokens 64
    python 03_kv_cache_demo.py --model gpt2-medium

依赖:
    pip install torch transformers
"""

import argparse
import sys
import time

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("安装: pip install torch transformers")
    sys.exit(1)


def get_device():
    """获取可用设备。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"使用 GPU: {name} ({mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("使用 CPU（速度会慢很多，建议使用 GPU）")
    return device


def generate_without_kv_cache(
    model, input_ids: torch.Tensor, gen_tokens: int
) -> tuple[list[int], float]:
    """
    不使用 KV Cache 生成。
    每一步把完整的序列（包括已生成的 token）全部送进模型。
    """
    generated = input_ids.clone()
    total_time = 0.0

    for i in range(gen_tokens):
        start = time.perf_counter()

        with torch.no_grad():
            # 把完整序列送入模型，不使用 cache
            outputs = model(generated, use_cache=False)

        # 取最后一个位置的 logits，贪心选择
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        elapsed = time.perf_counter() - start
        total_time += elapsed

        if (i + 1) % 10 == 0:
            curr_len = generated.shape[1]
            print(
                f"  [无 KV Cache] 已生成 {i + 1}/{gen_tokens} tokens, "
                f"当前序列长度 {curr_len}, 本步耗时 {elapsed * 1000:.1f}ms"
            )

    tokens = generated[0].tolist()[input_ids.shape[1] :]
    return tokens, total_time


def generate_with_kv_cache(
    model, input_ids: torch.Tensor, gen_tokens: int
) -> tuple[list[int], float]:
    """
    使用 KV Cache 生成。
    第一步（prefill）处理完整输入，之后每步只送入新生成的 1 个 token。
    """
    total_time = 0.0
    generated_tokens = []
    past_key_values = None
    current_input = input_ids

    for i in range(gen_tokens):
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model(current_input, past_key_values=past_key_values, use_cache=True)

        # 更新 KV Cache
        past_key_values = outputs.past_key_values

        # 取最后一个位置的 logits
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())

        # 下一步只输入新生成的 token（KV Cache 已有之前的信息）
        current_input = next_token

        elapsed = time.perf_counter() - start
        total_time += elapsed

        if (i + 1) % 10 == 0:
            print(
                f"  [有 KV Cache] 已生成 {i + 1}/{gen_tokens} tokens, "
                f"本步耗时 {elapsed * 1000:.1f}ms"
            )

    return generated_tokens, total_time


def calculate_kv_cache_size(model, past_key_values) -> float:
    """计算 KV Cache 的显存占用（MB）。"""
    if past_key_values is None:
        return 0.0

    total_bytes = 0
    for layer_kv in past_key_values:
        for tensor in layer_kv:
            total_bytes += tensor.nelement() * tensor.element_size()

    return total_bytes / 1024 / 1024


def main():
    parser = argparse.ArgumentParser(
        description="对比有无 KV Cache 的推理速度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace 模型名 (default: gpt2)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="输入 prompt",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="输入序列长度，不足的用 padding 填充 (default: 128)",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=32,
        help="生成的 token 数量 (default: 32)",
    )
    args = parser.parse_args()

    device = get_device()

    # 加载模型
    print(f"\n加载模型: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    # 准备输入
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    actual_len = input_ids.shape[1]

    # 如果指定了 seq_length 且大于实际长度，用重复的 prompt 填充
    if args.seq_length > actual_len:
        repeats = (args.seq_length // actual_len) + 1
        input_ids = input_ids.repeat(1, repeats)[:, : args.seq_length]

    seq_len = input_ids.shape[1]
    print(f"输入序列长度: {seq_len} tokens")
    print(f"将生成: {args.gen_tokens} tokens")

    # 模型信息
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.1f}M")

    # Warmup
    print("\nWarmup ...")
    with torch.no_grad():
        _ = model(input_ids[:, :8], use_cache=False)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # 1. 不用 KV Cache
    print(f"\n{'='*60}")
    print("测试 1: 不使用 KV Cache")
    print(f"{'='*60}")

    tokens_no_cache, time_no_cache = generate_without_kv_cache(
        model, input_ids, args.gen_tokens
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    # 2. 用 KV Cache
    print(f"\n{'='*60}")
    print("测试 2: 使用 KV Cache")
    print(f"{'='*60}")

    tokens_with_cache, time_with_cache = generate_with_kv_cache(
        model, input_ids, args.gen_tokens
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    # 验证结果一致
    results_match = tokens_no_cache == tokens_with_cache

    # 结果对比
    print(f"\n{'='*60}")
    print("结果对比")
    print(f"{'='*60}")
    print(f"  输入长度:          {seq_len} tokens")
    print(f"  生成长度:          {args.gen_tokens} tokens")
    print()
    print(f"  无 KV Cache 总耗时: {time_no_cache * 1000:.1f} ms")
    print(
        f"  无 KV Cache 每 token: {time_no_cache / args.gen_tokens * 1000:.1f} ms/token"
    )
    print()
    print(f"  有 KV Cache 总耗时: {time_with_cache * 1000:.1f} ms")
    print(
        f"  有 KV Cache 每 token: {time_with_cache / args.gen_tokens * 1000:.1f} ms/token"
    )
    print()
    print(f"  加速比:            {time_no_cache / time_with_cache:.2f}x")
    print(f"  生成结果一致:      {'是' if results_match else '否（精度差异）'}")

    # 解码生成的文本
    decoded = tokenizer.decode(tokens_with_cache, skip_special_tokens=True)
    print(f"\n  生成文本: {decoded}")

    # 解释
    print(f"\n{'='*60}")
    print("原理说明")
    print(f"{'='*60}")
    print(f"  无 KV Cache 时，每生成第 i 个 token，需要处理长度为 (n+i) 的完整序列。")
    print(f"  总计算量 ∝ Σ(n+i)² for i=1..{args.gen_tokens}")
    print()
    print(f"  有 KV Cache 时，Prefill 一次性处理长度 n 的输入，")
    print(f"  之后每步只处理 1 个新 token（但需要和缓存的 KV 做 attention）。")
    print()
    print(f"  序列越长、生成越多 token，KV Cache 的加速比越明显。")
    print(f"  代价是 KV Cache 需要额外显存。")


if __name__ == "__main__":
    main()
