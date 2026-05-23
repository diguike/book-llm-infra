#!/usr/bin/env python3
"""
计算不同模型和精度下的显存需求。

包括：模型权重、KV Cache、激活值的显存计算。
帮助你在部署前评估需要什么规格的 GPU。

用法:
    python 02_memory_calc.py
    python 02_memory_calc.py --params 7 --precision fp16 --seq-length 4096 --batch-size 8
    python 02_memory_calc.py --model llama2-70b --seq-length 8192 --batch-size 16

依赖:
    无（纯 Python）
"""

import argparse
import sys


# 预定义模型配置
MODEL_CONFIGS = {
    "gpt2": {
        "name": "GPT-2",
        "params_b": 0.124,
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 12,
        "head_dim": 64,
        "hidden_dim": 768,
    },
    "llama2-7b": {
        "name": "Llama 2 7B",
        "params_b": 6.7,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 32,
        "head_dim": 128,
        "hidden_dim": 4096,
    },
    "llama2-13b": {
        "name": "Llama 2 13B",
        "params_b": 13.0,
        "n_layers": 40,
        "n_heads": 40,
        "n_kv_heads": 40,
        "head_dim": 128,
        "hidden_dim": 5120,
    },
    "llama2-70b": {
        "name": "Llama 2 70B",
        "params_b": 70.0,
        "n_layers": 80,
        "n_heads": 64,
        "n_kv_heads": 8,  # GQA
        "head_dim": 128,
        "hidden_dim": 8192,
    },
    "llama3-8b": {
        "name": "Llama 3 8B",
        "params_b": 8.0,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,  # GQA
        "head_dim": 128,
        "hidden_dim": 4096,
    },
    "llama3-70b": {
        "name": "Llama 3 70B",
        "params_b": 70.0,
        "n_layers": 80,
        "n_heads": 64,
        "n_kv_heads": 8,  # GQA
        "head_dim": 128,
        "hidden_dim": 8192,
    },
    "qwen2-7b": {
        "name": "Qwen2 7B",
        "params_b": 7.6,
        "n_layers": 28,
        "n_heads": 28,
        "n_kv_heads": 4,  # GQA
        "head_dim": 128,
        "hidden_dim": 3584,
    },
    "mistral-7b": {
        "name": "Mistral 7B",
        "params_b": 7.2,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,  # GQA
        "head_dim": 128,
        "hidden_dim": 4096,
    },
}

PRECISION_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
    "fp8": 1,
}

GPU_SPECS = {
    "RTX 3090": {"vram_gb": 24, "bandwidth_tb": 0.936},
    "RTX 4090": {"vram_gb": 24, "bandwidth_tb": 1.008},
    "A10": {"vram_gb": 24, "bandwidth_tb": 0.600},
    "A100 40GB": {"vram_gb": 40, "bandwidth_tb": 2.0},
    "A100 80GB": {"vram_gb": 80, "bandwidth_tb": 2.0},
    "H100 80GB": {"vram_gb": 80, "bandwidth_tb": 3.35},
    "H200": {"vram_gb": 141, "bandwidth_tb": 4.8},
    "L40S": {"vram_gb": 48, "bandwidth_tb": 0.864},
}


def calc_model_weight_memory(params_b: float, precision: str) -> float:
    """计算模型权重占用的显存（GB）。"""
    bytes_per_param = PRECISION_BYTES[precision]
    return params_b * bytes_per_param


def calc_kv_cache_memory(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_length: int,
    batch_size: int,
    precision: str,
) -> float:
    """
    计算 KV Cache 的显存占用（GB）。

    公式: 2 × n_layers × n_kv_heads × head_dim × seq_length × batch_size × bytes_per_param
    """
    bytes_per_param = PRECISION_BYTES[precision]
    # 2 = K + V 各一份
    total_bytes = (
        2 * n_layers * n_kv_heads * head_dim * seq_length * batch_size * bytes_per_param
    )
    return total_bytes / (1024**3)


def calc_activation_memory(
    hidden_dim: int, seq_length: int, batch_size: int, precision: str
) -> float:
    """
    粗略估算前向传播中激活值的显存占用（GB）。

    实际激活值显存取决于实现细节，这里给出数量级估算。
    主要来源: attention scores [batch, heads, seq, seq] + FFN 中间结果
    """
    bytes_per_param = PRECISION_BYTES.get(precision, 2)
    # 简化估算: ~10 × batch × seq × hidden × bytes
    total_bytes = 10 * batch_size * seq_length * hidden_dim * bytes_per_param
    return total_bytes / (1024**3)


def format_gb(value: float) -> str:
    """格式化 GB 值。"""
    if value < 0.1:
        return f"{value * 1024:.1f} MB"
    return f"{value:.2f} GB"


def analyze_model(
    model_key: str = None,
    params_b: float = None,
    n_layers: int = 32,
    n_heads: int = 32,
    n_kv_heads: int = None,
    head_dim: int = 128,
    hidden_dim: int = 4096,
    precision: str = "fp16",
    seq_length: int = 2048,
    batch_size: int = 1,
):
    """分析模型的显存需求。"""
    # 使用预定义模型配置
    if model_key and model_key in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_key]
        params_b = config["params_b"]
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]
        n_kv_heads = config["n_kv_heads"]
        head_dim = config["head_dim"]
        hidden_dim = config["hidden_dim"]
        name = config["name"]
    else:
        if params_b is None:
            params_b = 7.0
        name = f"Custom {params_b}B"

    if n_kv_heads is None:
        n_kv_heads = n_heads

    print("=" * 60)
    print(f"显存需求分析: {name}")
    print("=" * 60)

    print(f"\n  模型配置:")
    print(f"    参数量:      {params_b}B")
    print(f"    层数:        {n_layers}")
    print(f"    注意力头数:  {n_heads} (KV heads: {n_kv_heads})")
    print(f"    Head 维度:   {head_dim}")
    print(f"    Hidden 维度: {hidden_dim}")
    print(f"    GQA:         {'是' if n_kv_heads < n_heads else '否'} (KV 头比例: {n_kv_heads}/{n_heads})")

    print(f"\n  运行配置:")
    print(f"    精度:        {precision}")
    print(f"    序列长度:    {seq_length}")
    print(f"    Batch size:  {batch_size}")

    # 计算各部分显存
    weight_mem = calc_model_weight_memory(params_b, precision)
    kv_cache_mem = calc_kv_cache_memory(
        n_layers, n_kv_heads, head_dim, seq_length, batch_size, precision
    )
    activation_mem = calc_activation_memory(
        hidden_dim, seq_length, batch_size, precision
    )
    cuda_overhead = 1.0  # 约 1GB CUDA context

    total_mem = weight_mem + kv_cache_mem + activation_mem + cuda_overhead

    print(f"\n  显存使用分解:")
    print(f"    模型权重:    {format_gb(weight_mem):>10}  ({weight_mem / total_mem * 100:.1f}%)")
    print(f"    KV Cache:    {format_gb(kv_cache_mem):>10}  ({kv_cache_mem / total_mem * 100:.1f}%)")
    print(f"    激活值:      {format_gb(activation_mem):>10}  ({activation_mem / total_mem * 100:.1f}%)")
    print(f"    CUDA 开销:   {format_gb(cuda_overhead):>10}  ({cuda_overhead / total_mem * 100:.1f}%)")
    print(f"    {'─' * 38}")
    print(f"    总计:        {format_gb(total_mem):>10}")

    # 如果用 MHA (全头) 对比
    if n_kv_heads < n_heads:
        kv_cache_mha = calc_kv_cache_memory(
            n_layers, n_heads, head_dim, seq_length, batch_size, precision
        )
        print(f"\n  GQA 对比:")
        print(f"    KV Cache (GQA, {n_kv_heads} heads): {format_gb(kv_cache_mem)}")
        print(f"    KV Cache (MHA, {n_heads} heads):  {format_gb(kv_cache_mha)}")
        print(f"    节省: {format_gb(kv_cache_mha - kv_cache_mem)} ({(1 - kv_cache_mem / kv_cache_mha) * 100:.0f}%)")

    # GPU 适配建议
    print(f"\n  GPU 适配建议:")
    for gpu_name, specs in GPU_SPECS.items():
        fits = total_mem <= specs["vram_gb"] * 0.9  # 留 10% margin
        status = "OK" if fits else "不够"
        utilization = total_mem / specs["vram_gb"] * 100

        # Decode 速度估算 (memory-bound)
        # 每 token 需要读取: 模型权重 + KV Cache (整个)
        read_per_token_gb = weight_mem + kv_cache_mem
        tokens_per_sec = specs["bandwidth_tb"] * 1024 / read_per_token_gb if read_per_token_gb > 0 else 0

        print(
            f"    {gpu_name:<14} ({specs['vram_gb']:>3}GB): "
            f"[{status:>3}] 占用 {utilization:.0f}%"
            f"  | Decode ~{tokens_per_sec:.0f} tok/s (理论上限)"
        )

    print()
    return total_mem


def compare_all_precisions(model_key: str, seq_length: int, batch_size: int):
    """对比同一模型在不同精度下的显存需求。"""
    if model_key not in MODEL_CONFIGS:
        print(f"未知模型: {model_key}")
        return

    config = MODEL_CONFIGS[model_key]
    print("=" * 60)
    print(f"精度对比: {config['name']} (seq={seq_length}, batch={batch_size})")
    print("=" * 60)

    header = f"  {'精度':<10} {'权重':>8} {'KV Cache':>10} {'总计':>10} {'适用GPU':>15}"
    print(header)
    print("  " + "-" * 58)

    for prec in ["fp32", "fp16", "int8", "int4"]:
        weight = calc_model_weight_memory(config["params_b"], prec)
        kv = calc_kv_cache_memory(
            config["n_layers"],
            config["n_kv_heads"],
            config["head_dim"],
            seq_length,
            batch_size,
            prec,
        )
        act = calc_activation_memory(config["hidden_dim"], seq_length, batch_size, prec)
        total = weight + kv + act + 1.0

        # 找最小能装下的 GPU
        fit_gpu = "多卡并行"
        for gpu_name, specs in GPU_SPECS.items():
            if total <= specs["vram_gb"] * 0.9:
                fit_gpu = gpu_name
                break

        print(
            f"  {prec:<10} {format_gb(weight):>8} {format_gb(kv):>10} {format_gb(total):>10} {fit_gpu:>15}"
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="计算 LLM 显存需求",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用的预定义模型: {', '.join(MODEL_CONFIGS.keys())}

示例:
  python 02_memory_calc.py --model llama2-7b
  python 02_memory_calc.py --model llama3-70b --seq-length 8192 --batch-size 16
  python 02_memory_calc.py --params 7 --precision int4
  python 02_memory_calc.py --compare llama2-7b
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"预定义模型名 ({', '.join(MODEL_CONFIGS.keys())})",
    )
    parser.add_argument("--params", type=float, default=None, help="模型参数量（B）")
    parser.add_argument("--n-layers", type=int, default=32, help="层数 (default: 32)")
    parser.add_argument("--n-heads", type=int, default=32, help="注意力头数 (default: 32)")
    parser.add_argument("--n-kv-heads", type=int, default=None, help="KV 头数，不指定则等于 n-heads")
    parser.add_argument("--head-dim", type=int, default=128, help="每头维度 (default: 128)")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="隐藏层维度 (default: 4096)")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=list(PRECISION_BYTES.keys()),
        help="精度 (default: fp16)",
    )
    parser.add_argument("--seq-length", type=int, default=2048, help="序列长度 (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--compare", type=str, default=None, help="对比不同精度的显存需求")

    args = parser.parse_args()

    if args.compare:
        compare_all_precisions(args.compare, args.seq_length, args.batch_size)
        return

    if args.model is None and args.params is None:
        # 默认展示几个常见模型
        print("未指定模型，展示常见模型的显存需求:\n")
        for model_key in ["llama2-7b", "llama3-8b", "llama2-70b"]:
            analyze_model(
                model_key=model_key,
                precision=args.precision,
                seq_length=args.seq_length,
                batch_size=args.batch_size,
            )

        # 对比 Llama 2 7B 不同精度
        compare_all_precisions("llama2-7b", args.seq_length, args.batch_size)
    else:
        analyze_model(
            model_key=args.model,
            params_b=args.params,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            hidden_dim=args.hidden_dim,
            precision=args.precision,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
