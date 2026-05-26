"""用 GPTQModel 把模型量化为 GPTQ INT4

使用方法:
    # 量化 Qwen2.5-7B 到 INT4
    python 02_gptq_quantize.py --model Qwen/Qwen2.5-7B-Instruct --bits 4 --output ./qwen2.5-7b-gptq-int4

硬件要求: NVIDIA GPU 24 GB+（量化过程要加载完整 FP16 模型）

依赖:
    pip install gptqmodel datasets

历史背景:
    原 AutoGPTQ 在 2025 年 4 月归档，社区已迁移到 GPTQModel（ModelCloud 维护）。
    GPTQModel 同时支持 GPTQ / AWQ / GGUF / FP8 多种量化方法，本脚本只演示 GPTQ。
"""
import argparse
import time


def get_calibration_dataset(tokenizer, n_samples: int = 1024):
    """从 C4 加载校准样本。

    校准（calibration）= 跑 N 条文本前向、收集每层激活 X、算 H = X^T X。
    经验：128 ~ 1024 条够用，多了边际收益递减。
    服务垂直领域时把领域文本混进来效果更好。
    """
    from datasets import load_dataset

    print(f"准备校准数据: {n_samples} 条（来自 C4 英文网页语料）")
    return [
        tokenizer(example["text"])
        for example in load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train",
        ).select(range(n_samples))
    ]


def quantize_gptq(model_id: str, bits: int, output_dir: str, n_samples: int):
    """GPTQ 量化主流程"""
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError:
        print("请先安装 gptqmodel: pip install gptqmodel")
        print("（注意：旧的 auto-gptq 已归档，本脚本用 gptqmodel）")
        return

    from transformers import AutoTokenizer

    print(f"开始 GPTQ 量化: {model_id} -> INT{bits}")
    print(f"输出目录: {output_dir}")

    # 1. 配置量化参数
    quant_config = QuantizeConfig(
        bits=bits,          # 量化位数。4 是精度/压缩的最佳平衡；3-bit 损失明显，8-bit 收益小
        group_size=128,     # 每 128 个权重共享一组 scale/zero-point；越小越精确，元数据开销越大
        desc_act=False,     # True 按激活幅度重排顺序，精度↑ 速度↓，一般 False 足够
        damp_percent=0.01,  # Hessian 阻尼系数，矩阵奇异时加这点对角线防数值爆炸
        sym=True,           # 对称量化（无 zero-point），kernel 更快、精度略低
    )
    print(f"\n量化配置: bits={bits}, group_size=128, desc_act=False, sym=True")

    # 2. 加载 tokenizer（校准要把文本切成 token id，必须和待量化模型严格匹配）
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 3. 加载待量化模型（GPTQModel.load 内部用 transformers 加载 FP16 权重，再包一层量化器）
    print(f"\n加载模型: {model_id} ...")
    start = time.time()
    model = GPTQModel.load(model_id, quant_config)
    print(f"模型加载完成, 耗时 {time.time() - start:.1f}s")

    # 4. 准备校准数据
    calibration_dataset = get_calibration_dataset(tokenizer, n_samples)

    # 5. 执行量化
    #    流程：逐层跑前向收集激活 → 算 Hessian → 量化该层权重 → 进入下一层
    #    资源：A100 上 7B 模型大约 1-2 小时；峰值显存 ≈ FP16 模型大小 + 校准 batch
    print(f"\n开始量化（A100 上 7B 模型大约 1-2 小时）...")
    start = time.time()
    model.quantize(calibration_dataset)
    print(f"量化完成, 耗时 {time.time() - start:.1f}s")

    # 6. 保存
    #    产物：safetensors（INT4 权重）+ quantize_config.json + tokenizer 文件
    #    可被 vLLM / SGLang / transformers 直接加载，无需额外转换
    print(f"\n保存量化模型到 {output_dir} ...")
    model.save(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("保存完成")


def main():
    parser = argparse.ArgumentParser(description="GPTQ 模型量化（基于 GPTQModel）")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace 模型名称或本地路径")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8],
                        help="量化位数")
    parser.add_argument("--output", type=str, default="./quantized-model-gptq",
                        help="输出目录")
    parser.add_argument("--samples", type=int, default=1024,
                        help="校准样本数（128 ~ 1024，多了边际收益递减）")
    args = parser.parse_args()
    quantize_gptq(args.model, args.bits, args.output, args.samples)


if __name__ == "__main__":
    main()
