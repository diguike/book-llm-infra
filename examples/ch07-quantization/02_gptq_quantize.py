"""用 AutoGPTQ 量化模型

使用方法:
    # 量化 Qwen2-7B 到 INT4
    python 02_gptq_quantize.py --model Qwen/Qwen2-7B --bits 4 --output ./qwen2-7b-gptq-int4

硬件要求: GPU 24GB+ (量化过程需要加载完整模型)
"""
import argparse
import time
import torch
from transformers import AutoTokenizer

def get_calibration_data(tokenizer, n_samples=128, seq_len=2048):
    """生成校准数据集（实际使用时应用真实数据）"""
    print(f"准备校准数据: {n_samples} 条, 长度 {seq_len}")

    # 这里用简单的中文文本作为示例
    # 实际量化时建议用 C4 或领域相关数据
    sample_texts = [
        "大语言模型的推理优化是一个重要的研究方向。量化技术可以显著降低模型的显存占用和推理延迟。",
        "Transformer 架构由 Self-Attention 和 Feed-Forward Network 组成。每一层都会对输入进行非线性变换。",
        "KV Cache 是推理加速的核心技术。它缓存了之前 token 的 Key 和 Value 向量，避免重复计算。",
        "PagedAttention 借鉴了操作系统虚拟内存的思想，将 KV Cache 分页管理，大幅提升显存利用率。",
    ] * (n_samples // 4 + 1)

    calibration_data = []
    for text in sample_texts[:n_samples]:
        tokens = tokenizer(text, return_tensors="pt", padding="max_length",
                          max_length=seq_len, truncation=True)
        calibration_data.append(tokens.input_ids)

    return calibration_data

def quantize_gptq(model_name: str, bits: int, output_dir: str):
    """GPTQ 量化流程"""
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        print("请先安装 auto-gptq: pip install auto-gptq")
        print("注意: auto-gptq 需要 CUDA 环境")
        return

    print(f"开始 GPTQ 量化: {model_name} -> INT{bits}")
    print(f"输出目录: {output_dir}")

    # 1. 配置量化参数
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,    # 每 128 个权重共享一个 scale/zero_point
        desc_act=False,    # True 精度更好但更慢，一般用 False
        damp_percent=0.1,  # Hessian 阻尼系数
    )

    print(f"\n量化配置:")
    print(f"  bits: {bits}")
    print(f"  group_size: 128")
    print(f"  desc_act: False")

    # 2. 加载模型
    print(f"\n加载模型: {model_name} ...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        trust_remote_code=True,
    )
    print(f"模型加载完成, 耗时 {time.time() - start:.1f}s")

    # 3. 准备校准数据
    calibration_data = get_calibration_data(tokenizer)

    # 4. 执行量化
    print(f"\n开始量化 (这可能需要 10-30 分钟) ...")
    start = time.time()
    model.quantize(calibration_data)
    print(f"量化完成, 耗时 {time.time() - start:.1f}s")

    # 5. 保存
    print(f"\n保存量化模型到 {output_dir} ...")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("保存完成")

    # 6. 验证
    print(f"\n验证量化模型 ...")
    model = AutoGPTQForCausalLM.from_quantized(
        output_dir, device="cuda:0", trust_remote_code=True
    )
    inputs = tokenizer("量化后的模型", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    print(f"生成测试: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

def main():
    parser = argparse.ArgumentParser(description="GPTQ 模型量化")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B",
                       help="HuggingFace 模型名称或本地路径")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8],
                       help="量化位数")
    parser.add_argument("--output", type=str, default="./quantized-model-gptq",
                       help="输出目录")
    parser.add_argument("--samples", type=int, default=128,
                       help="校准数据条数")
    args = parser.parse_args()
    quantize_gptq(args.model, args.bits, args.output)

if __name__ == "__main__":
    main()
