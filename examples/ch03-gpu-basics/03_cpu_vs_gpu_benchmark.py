#!/usr/bin/env python3
"""
矩阵运算 CPU vs GPU 性能对比。

运行不同大小的矩阵乘法，对比 CPU 和 GPU 的执行时间。
矩阵乘法是 Transformer 的核心操作，这个 benchmark 能直观感受 GPU 加速的效果。

用法:
    python 03_cpu_vs_gpu_benchmark.py
    python 03_cpu_vs_gpu_benchmark.py --sizes 512 1024 2048 4096
    python 03_cpu_vs_gpu_benchmark.py --dtype float16 --warmup 5 --repeat 20

依赖:
    pip install torch
"""

import argparse
import sys
import time

try:
    import torch
except ImportError:
    print("缺少依赖: pip install torch")
    sys.exit(1)


def benchmark_matmul(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int = 3,
    repeat: int = 10,
) -> float:
    """
    执行矩阵乘法并计时。

    Returns:
        平均耗时（ms）
    """
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup):
        _ = torch.mm(a, b)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = torch.mm(a, b)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    # 去掉最高和最低，取平均
    times.sort()
    if len(times) > 4:
        times = times[1:-1]

    return sum(times) / len(times)


def calc_flops(size: int) -> float:
    """计算矩阵乘法的 FLOPs。[M,K] @ [K,N] = 2*M*N*K FLOPs。"""
    return 2.0 * size * size * size


def format_tflops(flops: float, time_ms: float) -> str:
    """将 FLOPs 和时间转换为 TFLOPS。"""
    tflops = flops / (time_ms / 1000) / 1e12
    return f"{tflops:.2f}"


def run_benchmark(args):
    """运行完整 benchmark。"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    has_gpu = torch.cuda.is_available()

    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("未检测到 GPU，只运行 CPU benchmark")

    print(f"精度: {args.dtype}")
    print(f"Warmup: {args.warmup} 次, Repeat: {args.repeat} 次")
    print()

    # CPU 不支持 float16/bfloat16 的高效计算，用 float32
    cpu_dtype = torch.float32

    # 表头
    header = (
        f"  {'矩阵大小':>10} {'FLOPs':>10} "
        f"{'CPU (ms)':>10} {'CPU TFLOPS':>11}"
    )
    if has_gpu:
        header += f" {'GPU (ms)':>10} {'GPU TFLOPS':>11} {'加速比':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []

    for size in args.sizes:
        flops = calc_flops(size)
        flops_str = f"{flops / 1e9:.1f}G"

        # CPU benchmark
        try:
            cpu_time = benchmark_matmul(
                size, torch.device("cpu"), cpu_dtype, args.warmup, args.repeat
            )
            cpu_tflops = format_tflops(flops, cpu_time)
        except Exception as e:
            cpu_time = float("inf")
            cpu_tflops = "N/A"

        row = f"  {size:>5}x{size:<5} {flops_str:>10} {cpu_time:>9.2f}ms {cpu_tflops:>10} TFLOPS"

        # GPU benchmark
        if has_gpu:
            try:
                gpu_time = benchmark_matmul(
                    size, torch.device("cuda"), dtype, args.warmup, args.repeat
                )
                gpu_tflops = format_tflops(flops, gpu_time)
                speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
                row += f" {gpu_time:>9.2f}ms {gpu_tflops:>10} TFLOPS {speedup:>7.1f}x"
            except Exception as e:
                row += f" {'ERROR':>10} {'N/A':>11} {'N/A':>8}"
                gpu_time = None
                speedup = None

            results.append(
                {
                    "size": size,
                    "cpu_ms": cpu_time,
                    "gpu_ms": gpu_time,
                    "speedup": speedup,
                }
            )
        else:
            results.append({"size": size, "cpu_ms": cpu_time})

        print(row)

    # 总结
    print()
    print("=" * 60)
    print("分析")
    print("=" * 60)

    if has_gpu and results:
        speedups = [r["speedup"] for r in results if r.get("speedup")]
        if speedups:
            print(f"  最小加速比: {min(speedups):.1f}x (矩阵较小时)")
            print(f"  最大加速比: {max(speedups):.1f}x (矩阵较大时)")
            print()

    print("  关键发现:")
    print("  1. 矩阵越大，GPU 的加速比越明显")
    print("     小矩阵时 GPU 的启动开销（kernel launch）占比大")
    print("     大矩阵时 GPU 的并行能力充分发挥")
    print()
    print("  2. 对应 LLM 推理:")
    print("     Prefill 阶段: 大矩阵乘法 → GPU 加速比高")
    print("     Decode 阶段: 向量×矩阵（batch=1时）→ 加速比较低，受限于显存带宽")
    print()

    if has_gpu and dtype == torch.float16:
        print("  3. FP16 在 GPU 上比 FP32 更快:")
        print("     Tensor Core 支持 FP16 原生加速")
        print("     显存带宽减半 → 读写速度翻倍")
        print("     这就是推理时用 FP16/BF16 的原因")
        print()


def run_transformer_simulation(args):
    """模拟 Transformer 单层的计算，对比 CPU/GPU。"""
    if not torch.cuda.is_available():
        print("需要 GPU 来运行 Transformer 模拟")
        return

    print("\n" + "=" * 60)
    print("Transformer 单层计算模拟 (Llama 2 7B 参数)")
    print("=" * 60)

    hidden_dim = 4096
    intermediate_dim = 11008
    n_heads = 32
    head_dim = 128
    seq_lengths = [128, 512, 2048]
    batch_size = 1

    dtype = torch.float16
    device = torch.device("cuda")

    print(f"  hidden_dim={hidden_dim}, n_heads={n_heads}, batch={batch_size}, dtype=fp16")
    print()

    header = f"  {'seq_len':>8} {'QKV proj':>10} {'Attention':>10} {'FFN':>10} {'Total':>10}"
    print(header)
    print("  " + "-" * 52)

    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

        # QKV Projection: [B, T, 4096] @ [4096, 4096*3]
        wqkv = torch.randn(hidden_dim, hidden_dim * 3, device=device, dtype=dtype)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            qkv = x.view(-1, hidden_dim) @ wqkv
        torch.cuda.synchronize()
        qkv_time = (time.perf_counter() - start) / 10 * 1000

        # Attention: Q@K^T + softmax + @V (simplified)
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
            weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(weights, v)
        torch.cuda.synchronize()
        attn_time = (time.perf_counter() - start) / 10 * 1000

        # FFN: [B*T, 4096] @ [4096, 11008] + [B*T, 11008] @ [11008, 4096]
        w1 = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype)
        w2 = torch.randn(intermediate_dim, hidden_dim, device=device, dtype=dtype)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            h = x.view(-1, hidden_dim) @ w1
            h = torch.nn.functional.silu(h)
            h = h @ w2
        torch.cuda.synchronize()
        ffn_time = (time.perf_counter() - start) / 10 * 1000

        total = qkv_time + attn_time + ffn_time
        print(
            f"  {seq_len:>8} {qkv_time:>9.2f}ms {attn_time:>9.2f}ms "
            f"{ffn_time:>9.2f}ms {total:>9.2f}ms"
        )

    print()
    print("  观察:")
    print("  - QKV 和 FFN 时间随 seq_len 线性增长（矩阵乘法）")
    print("  - Attention 时间随 seq_len 平方增长（Q@K^T 的 O(n^2)）")
    print("  - 短序列时 FFN 占主导，长序列时 Attention 占主导")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="CPU vs GPU 矩阵运算性能对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096],
        help="矩阵大小列表 (default: 256 512 1024 2048 4096)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="GPU 计算精度 (default: float32)",
    )
    parser.add_argument("--warmup", type=int, default=3, help="预热次数 (default: 3)")
    parser.add_argument("--repeat", type=int, default=10, help="重复次数 (default: 10)")
    parser.add_argument(
        "--transformer-sim",
        action="store_true",
        help="运行 Transformer 单层计算模拟",
    )
    args = parser.parse_args()

    run_benchmark(args)

    if args.transformer_sim:
        run_transformer_simulation(args)


if __name__ == "__main__":
    main()
