"""演示不同精度的数值表示和精度损失"""
import torch
import argparse
import struct

def show_float_representation():
    """展示不同浮点格式的内部表示"""
    values = [3.14, 0.1, 1e-7, 65504.0, 1e38]

    print("=" * 80)
    print("不同精度的数值表示对比")
    print("=" * 80)

    for val in values:
        fp32 = torch.tensor(val, dtype=torch.float32)
        fp16 = torch.tensor(val, dtype=torch.float16)
        bf16 = torch.tensor(val, dtype=torch.bfloat16)

        print(f"\n原始值: {val}")
        print(f"  FP32: {fp32.item():.10f}  (4 bytes)")
        print(f"  FP16: {fp16.item():.10f}  (2 bytes)")
        print(f"  BF16: {bf16.item():.10f}  (2 bytes)")
        print(f"  FP16 误差: {abs(fp32.item() - fp16.item()):.2e}")
        print(f"  BF16 误差: {abs(fp32.item() - bf16.item()):.2e}")

def simulate_int_quantization():
    """模拟 INT8/INT4 量化过程"""
    print("\n" + "=" * 80)
    print("INT 量化模拟")
    print("=" * 80)

    # 模拟一组权重
    torch.manual_seed(42)
    weights = torch.randn(1000) * 0.5  # 典型的模型权重分布

    for bits in [8, 4]:
        n_levels = 2 ** bits
        w_min, w_max = weights.min(), weights.max()
        scale = (w_max - w_min) / (n_levels - 1)
        zero_point = (-w_min / scale).round().int()

        # 量化
        quantized = ((weights - w_min) / scale).round().clamp(0, n_levels - 1).int()
        # 反量化
        dequantized = quantized.float() * scale + w_min

        # 计算误差
        mse = ((weights - dequantized) ** 2).mean()
        max_err = (weights - dequantized).abs().max()

        print(f"\nINT{bits} 量化 ({n_levels} 级):")
        print(f"  Scale: {scale:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  最大误差: {max_err:.6f}")
        print(f"  压缩比: {32/bits:.1f}x")

def model_memory_calc():
    """计算不同精度下模型的显存占用"""
    print("\n" + "=" * 80)
    print("7B 模型在不同精度下的显存占用")
    print("=" * 80)

    params = 7e9
    precisions = {
        "FP32": 4,
        "FP16/BF16": 2,
        "INT8": 1,
        "INT4": 0.5,
    }

    print(f"\n模型参数量: {params/1e9:.0f}B")
    print(f"{'精度':<12} {'每参数字节':<12} {'模型大小':<12} {'推理显存估算':<15}")
    print("-" * 55)
    for name, bytes_per_param in precisions.items():
        model_size = params * bytes_per_param / (1024**3)
        # 推理时还需要 KV Cache 和激活值，大约额外 20-50%
        inference_mem = model_size * 1.3
        print(f"{name:<12} {bytes_per_param:<12} {model_size:.1f} GB{'':<5} ~{inference_mem:.1f} GB")

def main():
    parser = argparse.ArgumentParser(description="数值精度与量化演示")
    parser.add_argument("--all", action="store_true", help="运行所有演示")
    parser.add_argument("--float", action="store_true", help="浮点精度对比")
    parser.add_argument("--quant", action="store_true", help="INT 量化模拟")
    parser.add_argument("--memory", action="store_true", help="显存计算")
    args = parser.parse_args()

    if args.all or not any([args.float, args.quant, args.memory]):
        args.float = args.quant = args.memory = True

    if args.float:
        show_float_representation()
    if args.quant:
        simulate_int_quantization()
    if args.memory:
        model_memory_calc()

if __name__ == "__main__":
    main()
