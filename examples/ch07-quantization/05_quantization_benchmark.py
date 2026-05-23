"""量化前后的性能对比

使用方法:
    # 对比 FP16 和 INT4 推理
    python 05_quantization_benchmark.py --model Qwen/Qwen2-7B --quantized ./qwen2-7b-awq

硬件要求: GPU 16GB+ (INT4) 或 24GB+ (FP16 + INT4 对比)
"""
import argparse
import time
import torch

def benchmark_generation(model, tokenizer, prompts, max_new_tokens=128):
    """测量生成性能"""
    device = next(model.parameters()).device

    total_tokens = 0
    total_time = 0
    ttfts = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        # 预热
        with torch.no_grad():
            start = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            elapsed = time.perf_counter() - start

        gen_tokens = outputs.shape[1] - input_len
        total_tokens += gen_tokens
        total_time += elapsed

    return {
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_second": total_tokens / total_time,
    }

def get_model_memory(model):
    """获取模型显存占用"""
    mem = sum(p.nelement() * p.element_size() for p in model.parameters())
    return mem / (1024 ** 3)  # GB

def main():
    parser = argparse.ArgumentParser(description="量化性能对比")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B",
                       help="原始模型")
    parser.add_argument("--quantized", type=str, default=None,
                       help="量化后的模型路径 (GPTQ 或 AWQ)")
    parser.add_argument("--prompts", type=int, default=5,
                       help="测试 prompt 数量")
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="最大生成 token 数")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    test_prompts = [
        "请解释什么是 Transformer 架构",
        "Python 和 JavaScript 的主要区别是什么",
        "如何优化大语言模型的推理速度",
        "什么是 KV Cache，它为什么重要",
        "解释分布式训练中的数据并行和模型并行",
    ][:args.prompts]

    results = {}

    # 测试原始模型 (FP16)
    print("=" * 60)
    print("加载原始模型 (FP16) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    try:
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True
        )
        mem_fp16 = get_model_memory(model_fp16)
        print(f"模型显存: {mem_fp16:.2f} GB")

        print("开始 benchmark ...")
        results["FP16"] = benchmark_generation(
            model_fp16, tokenizer, test_prompts, args.max_tokens
        )
        results["FP16"]["memory_gb"] = mem_fp16

        del model_fp16
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"FP16 模型加载失败 (可能显存不足): {e}")

    # 测试量化模型
    if args.quantized:
        print(f"\n加载量化模型: {args.quantized} ...")
        try:
            from auto_gptq import AutoGPTQForCausalLM
            model_quant = AutoGPTQForCausalLM.from_quantized(
                args.quantized, device="cuda:0", trust_remote_code=True
            )
        except Exception:
            try:
                from awq import AutoAWQForCausalLM
                model_quant = AutoAWQForCausalLM.from_quantized(
                    args.quantized, fuse_layers=True, trust_remote_code=True
                )
            except Exception:
                model_quant = AutoModelForCausalLM.from_pretrained(
                    args.quantized, device_map="auto", trust_remote_code=True
                )

        mem_quant = get_model_memory(model_quant)
        print(f"模型显存: {mem_quant:.2f} GB")

        print("开始 benchmark ...")
        results["Quantized"] = benchmark_generation(
            model_quant, tokenizer, test_prompts, args.max_tokens
        )
        results["Quantized"]["memory_gb"] = mem_quant

    # 打印结果
    print("\n" + "=" * 60)
    print("性能对比结果")
    print("=" * 60)
    print(f"{'指标':<20} ", end="")
    for name in results:
        print(f"{name:<15} ", end="")
    print()
    print("-" * 55)

    for metric, label in [
        ("memory_gb", "显存占用 (GB)"),
        ("tokens_per_second", "生成速度 (tok/s)"),
        ("total_time", "总耗时 (s)"),
    ]:
        print(f"{label:<20} ", end="")
        for name in results:
            val = results[name].get(metric, 0)
            print(f"{val:<15.2f} ", end="")
        print()

if __name__ == "__main__":
    main()
