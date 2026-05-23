#!/usr/bin/env python3
"""
获取 GPU 信息和显存使用情况。

显示 CUDA 版本、GPU 型号、显存容量、当前使用情况等。
如果没有 GPU，会显示 CPU 信息作为参考。

用法:
    python 01_gpu_info.py
    python 01_gpu_info.py --allocate 1024  # 分配 1024MB 显存观察变化

依赖:
    pip install torch
"""

import argparse
import sys

try:
    import torch
except ImportError:
    print("缺少依赖: pip install torch")
    sys.exit(1)


def show_system_info():
    """显示系统和 PyTorch 信息。"""
    print("=" * 60)
    print("系统信息")
    print("=" * 60)
    print(f"  PyTorch 版本:     {torch.__version__}")
    print(f"  CUDA 可用:        {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA 版本:        {torch.version.cuda}")
        print(f"  cuDNN 版本:       {torch.backends.cudnn.version()}")
        print(f"  GPU 数量:         {torch.cuda.device_count()}")
    else:
        print("  (未检测到 GPU)")
    print()


def show_gpu_info(device_id: int = 0):
    """显示指定 GPU 的详细信息。"""
    if not torch.cuda.is_available():
        print("没有可用的 GPU")
        return

    props = torch.cuda.get_device_properties(device_id)

    print("=" * 60)
    print(f"GPU {device_id} 详细信息")
    print("=" * 60)
    print(f"  名称:              {props.name}")
    print(f"  计算能力:          {props.major}.{props.minor}")
    print(f"  SM 数量:           {props.multi_processor_count}")
    print(f"  总显存:            {props.total_mem / 1024**3:.2f} GB")

    # 当前显存使用
    mem_allocated = torch.cuda.memory_allocated(device_id)
    mem_reserved = torch.cuda.memory_reserved(device_id)
    mem_total = props.total_mem

    print()
    print("  显存使用:")
    print(f"    已分配 (allocated): {mem_allocated / 1024**2:.1f} MB")
    print(f"    已预留 (reserved):  {mem_reserved / 1024**2:.1f} MB")
    print(f"    总量 (total):       {mem_total / 1024**2:.1f} MB")
    print(f"    可用 (free):        {(mem_total - mem_reserved) / 1024**2:.1f} MB")

    # 能力信息
    print()
    print("  硬件能力:")
    print(f"    最大线程数/Block:   {props.max_threads_per_multi_processor}")

    # Tensor Core 支持判断
    compute_cap = props.major * 10 + props.minor
    if compute_cap >= 70:
        print(f"    Tensor Core:        支持 (compute capability >= 7.0)")
    else:
        print(f"    Tensor Core:        不支持 (需要 compute capability >= 7.0)")

    if compute_cap >= 80:
        print(f"    BF16:               支持")
    else:
        print(f"    BF16:               不支持 (需要 Ampere 架构+)")

    if compute_cap >= 89:
        print(f"    FP8:                支持")
    else:
        print(f"    FP8:                不支持 (需要 Hopper/Ada 架构)")

    print()


def show_model_memory_estimate():
    """估算常见模型的显存需求。"""
    print("=" * 60)
    print("常见模型显存需求估算（仅模型权重）")
    print("=" * 60)

    models = [
        ("GPT-2", 0.124),
        ("Llama 2 7B", 7.0),
        ("Llama 2 13B", 13.0),
        ("Llama 2 70B", 70.0),
        ("Llama 3 8B", 8.0),
        ("Llama 3 70B", 70.0),
        ("Qwen2 72B", 72.0),
    ]

    precisions = [
        ("FP32", 4),
        ("FP16/BF16", 2),
        ("INT8", 1),
        ("INT4", 0.5),
    ]

    # 表头
    header = f"  {'模型':<15}"
    for name, _ in precisions:
        header += f"  {name:>10}"
    print(header)
    print("  " + "-" * (15 + 12 * len(precisions)))

    for model_name, params_b in models:
        row = f"  {model_name:<15}"
        for _, bytes_per in precisions:
            mem_gb = params_b * bytes_per
            row += f"  {mem_gb:>8.1f} GB"
        print(row)

    print()
    print("  注意: 实际推理显存 = 模型权重 + KV Cache + 激活值 + CUDA overhead")
    print("  KV Cache 随 batch_size 和 seq_length 线性增长，可能占总显存的 30-70%")
    print()


def allocate_test(size_mb: int):
    """分配指定大小的 GPU 显存，观察显存变化。"""
    if not torch.cuda.is_available():
        print("没有可用的 GPU，跳过显存分配测试")
        return

    print("=" * 60)
    print(f"显存分配测试: 分配 {size_mb} MB")
    print("=" * 60)

    # 分配前
    before_alloc = torch.cuda.memory_allocated(0)
    before_reserved = torch.cuda.memory_reserved(0)
    print(f"  分配前: allocated={before_alloc / 1024**2:.1f} MB, reserved={before_reserved / 1024**2:.1f} MB")

    # 分配
    n_elements = size_mb * 1024 * 1024 // 4  # FP32, 4 bytes each
    try:
        tensor = torch.randn(n_elements, device="cuda", dtype=torch.float32)

        after_alloc = torch.cuda.memory_allocated(0)
        after_reserved = torch.cuda.memory_reserved(0)
        actual_mb = (after_alloc - before_alloc) / 1024**2

        print(f"  分配后: allocated={after_alloc / 1024**2:.1f} MB, reserved={after_reserved / 1024**2:.1f} MB")
        print(f"  实际增加: {actual_mb:.1f} MB (requested: {size_mb} MB)")
        print()
        print(f"  注意 reserved >= allocated:")
        print(f"    PyTorch 的 CUDA 内存分配器会预留一块较大的内存池，")
        print(f"    以避免频繁调用 cudaMalloc（很慢）。")
        print(f"    所以 reserved 通常大于 allocated。")

        # 释放
        del tensor
        torch.cuda.empty_cache()
        final_alloc = torch.cuda.memory_allocated(0)
        final_reserved = torch.cuda.memory_reserved(0)
        print(f"\n  释放后: allocated={final_alloc / 1024**2:.1f} MB, reserved={final_reserved / 1024**2:.1f} MB")
        print(f"  注意: empty_cache() 后 reserved 可能仍然不为 0（缓存池未完全归还 OS）")

    except RuntimeError as e:
        print(f"  分配失败: {e}")
        print(f"  这就是 CUDA OOM 的样子。解决方案: 减小 batch_size / 使用量化 / 换更大显存的 GPU")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="获取 GPU 信息和显存使用情况",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--allocate",
        type=int,
        default=0,
        help="分配指定大小的显存（MB）观察变化",
    )
    args = parser.parse_args()

    show_system_info()
    show_gpu_info()
    show_model_memory_estimate()

    if args.allocate > 0:
        allocate_test(args.allocate)


if __name__ == "__main__":
    main()
