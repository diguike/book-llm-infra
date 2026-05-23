"""对比标准 Attention 和 FlashAttention 的速度与显存

标准 Attention 显式物化 [n, n] 的 attention 矩阵，显存 O(n²)。
FlashAttention 用 tiling 分块计算，显存 O(n)，速度也更快。

使用方法:
    python 01_flash_attention_demo.py
    python 01_flash_attention_demo.py --seq-lengths 512 1024 2048 4096 8192

硬件要求: GPU (建议 8GB+, FlashAttention 需要 SM >= 80 即 A100/RTX 3090+)
"""

import argparse
import time
import torch
import torch.nn.functional as F


def naive_attention(q, k, v):
    """标准 Attention：显式计算完整的 [n, n] score 矩阵"""
    d_k = q.shape[-1]
    # [B, H, N, D] @ [B, H, D, N] → [B, H, N, N]  ← O(n²) 显存
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    # [B, H, N, N] @ [B, H, N, D] → [B, H, N, D]
    return torch.matmul(weights, v)


def flash_attention(q, k, v):
    """PyTorch 内置的 scaled_dot_product_attention

    自动选择最优后端：
    - FlashAttention (SM >= 80, FP16/BF16)
    - Memory-Efficient Attention (xformers)
    - 标准数学实现 (fallback)
    """
    return F.scaled_dot_product_attention(q, k, v)


def measure(fn, q, k, v, warmup=2, repeats=5):
    """测量函数的执行时间和峰值显存"""
    device = q.device

    # Warmup
    for _ in range(warmup):
        try:
            _ = fn(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return float("inf"), float("inf")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        try:
            out = fn(q, k, v)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return float("inf"), float("inf")
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / repeats * 1000

    peak_mb = 0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    del out
    return elapsed_ms, peak_mb


def run_benchmark(seq_lengths, n_heads=32, head_dim=128, batch_size=1):
    """对比不同 seq_length 下两种实现的性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if device.type == "cpu":
        print("⚠  未检测到 GPU。FlashAttention 需要 GPU (SM >= 80) 才能启用。")
        print("   CPU 上两种实现都退化为标准数学运算，对比意义有限。\n")

    print(f"配置: batch={batch_size}, heads={n_heads}, head_dim={head_dim}, dtype={dtype}")
    print()
    header = f"{'seq_len':<10} {'Naive (ms)':<14} {'Flash (ms)':<14} {'加速比':<10} {'Naive 显存':<14} {'Flash 显存':<14} {'显存节省':<10}"
    print(header)
    print("-" * len(header))

    for seq_len in seq_lengths:
        q = torch.randn(batch_size, n_heads, seq_len, head_dim,
                        device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        naive_ms, naive_mb = measure(naive_attention, q, k, v)
        # 清理后再测 flash
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        flash_ms, flash_mb = measure(flash_attention, q, k, v)

        # 格式化输出
        def fmt_ms(ms):
            return "OOM" if ms == float("inf") else f"{ms:.1f}"

        def fmt_mb(mb):
            return "OOM" if mb == float("inf") else f"{mb:.0f} MB"

        speedup = naive_ms / flash_ms if flash_ms > 0 and flash_ms != float("inf") else 0
        mem_save = (1 - flash_mb / naive_mb) if (naive_mb > 0 and naive_mb != float("inf") and flash_mb != float("inf")) else 0

        print(f"{seq_len:<10} {fmt_ms(naive_ms):<14} {fmt_ms(flash_ms):<14} "
              f"{speedup:.2f}x{'':<5} {fmt_mb(naive_mb):<14} {fmt_mb(flash_mb):<14} "
              f"{mem_save:.0%}")

        del q, k, v
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print("说明:")
    print("  - FlashAttention 在长序列上优势更大 (显存 O(n) vs O(n²))")
    print("  - 加速比和显存节省会随 seq_len 增长而增大")
    print("  - 'OOM' 表示该配置下标准 Attention 显存不够")


def main():
    parser = argparse.ArgumentParser(description="FlashAttention vs Naive Attention 对比")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192],
                        help="要测试的 sequence lengths")
    parser.add_argument("--heads", type=int, default=32,
                        help="Attention heads 数量")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="每个 head 的维度")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    print("=" * 80)
    print("FlashAttention vs Naive Attention 性能对比")
    print("=" * 80)
    print()
    run_benchmark(args.seq_lengths, args.heads, args.head_dim, args.batch)


if __name__ == "__main__":
    main()
