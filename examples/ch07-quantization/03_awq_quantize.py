"""用 AutoAWQ 量化模型

使用方法:
    python 03_awq_quantize.py --model Qwen/Qwen2-7B --output ./qwen2-7b-awq

硬件要求: GPU 24GB+
"""
import argparse
import time
import torch

def quantize_awq(model_name: str, output_dir: str):
    """AWQ 量化流程"""
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("请先安装 autoawq: pip install autoawq")
        return

    from transformers import AutoTokenizer

    print(f"开始 AWQ 量化: {model_name}")

    # AWQ 量化配置
    quant_config = {
        "zero_point": True,    # 使用 zero-point 量化
        "q_group_size": 128,   # group size
        "w_bit": 4,            # 4-bit 量化
        "version": "GEMM",     # GEMM kernel（推理更快）
    }

    print(f"量化配置: {quant_config}")

    # 加载模型
    print(f"\n加载模型 ...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, safetensors=True
    )
    print(f"加载完成, 耗时 {time.time() - start:.1f}s")

    # 执行量化
    print(f"\n开始量化 ...")
    start = time.time()
    model.quantize(tokenizer, quant_config=quant_config)
    print(f"量化完成, 耗时 {time.time() - start:.1f}s")

    # 保存
    print(f"\n保存到 {output_dir} ...")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("完成")

def compare_gptq_awq():
    """对比 GPTQ 和 AWQ 的特点"""
    print("=" * 70)
    print("GPTQ vs AWQ vs GGUF 对比")
    print("=" * 70)

    comparison = """
| 特性           | GPTQ              | AWQ               | GGUF (llama.cpp)    |
|----------------|--------------------|--------------------|---------------------|
| 量化算法       | Hessian 逐层优化    | 激活感知权重保护     | 多种 (Q4_K_M 等)    |
| 精度 (一般)    | 好                 | 更好               | 好                  |
| 量化速度       | 慢 (10-30 min)     | 较快 (5-15 min)    | 快 (几分钟)         |
| 推理框架       | vLLM, HF           | vLLM, HF           | llama.cpp, Ollama   |
| GPU 推理速度   | 快                 | 快                 | 较慢 (非原生 CUDA)   |
| CPU 推理       | 不支持             | 不支持             | 原生支持            |
| 校准数据       | 需要 (~128 条)     | 需要 (~128 条)     | 不需要              |
| 主要用途       | GPU 部署           | GPU 部署           | 本地/CPU/边缘部署    |
"""
    print(comparison)

def main():
    parser = argparse.ArgumentParser(description="AWQ 模型量化")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B")
    parser.add_argument("--output", type=str, default="./quantized-model-awq")
    parser.add_argument("--compare", action="store_true", help="显示量化方法对比")
    args = parser.parse_args()

    if args.compare:
        compare_gptq_awq()
    else:
        quantize_awq(args.model, args.output)

if __name__ == "__main__":
    main()
