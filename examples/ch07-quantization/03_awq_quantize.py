"""用 llm-compressor 把模型量化为 AWQ INT4

使用方法:
    python 03_awq_quantize.py --model Qwen/Qwen2.5-7B-Instruct --output ./qwen2.5-7b-awq-int4
    python 03_awq_quantize.py --compare    # 仅打印 GPTQ / AWQ / GGUF 对比表

硬件要求: NVIDIA GPU 24 GB+

依赖:
    pip install llmcompressor datasets transformers

历史背景:
    原 AutoAWQ 在 2025 年 5 月归档；本脚本用 vLLM 团队官方推荐的 llm-compressor。
    llm-compressor 同时支持 AWQ / GPTQ / FP8 / INT8，本脚本只演示 AWQ。

API 提示:
    若你的 llm-compressor 版本和这里 API 不一致，参考官方仓库 examples/awq/。
"""
import argparse


def quantize_awq(model_id: str, output_dir: str, n_samples: int = 512):
    """AWQ 量化主流程"""
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
    except ImportError:
        print("请先安装 llmcompressor: pip install llmcompressor")
        print("（注意：旧的 autoawq 已归档，本脚本用 llm-compressor）")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"开始 AWQ 量化: {model_id}")

    # 1. 加载模型 + tokenizer
    print(f"\n加载模型 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 2. 准备校准数据
    #    AWQ 也要校准——用来识别"激活幅度大的关键通道"，再为这些通道找最优 scale
    print(f"准备校准数据: {n_samples} 条（来自 C4 英文网页语料）")
    ds = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    ).select(range(n_samples))

    def preprocess(example):
        tokens = tokenizer(
            example["text"], max_length=2048, truncation=True, return_tensors="pt",
        )
        return {"input_ids": tokens.input_ids[0]}

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    # 3. 配置 AWQ recipe
    #    W4A16_ASYM = 4-bit 权重 / 16-bit 激活 / 非对称量化，最常用的 AWQ 配置
    #    AWQModifier 内部会：
    #      1) 跑校准数据收集激活分布
    #      2) 找出"显著权重"（对应激活幅度大的通道）
    #      3) 给每层做 per-channel scaling 让难量化的通道变好量化
    #      4) 把权重量化到 INT4
    recipe = AWQModifier(
        targets="Linear",         # 只量化 Linear 层
        scheme="W4A16_ASYM",
        ignore=["lm_head"],       # lm_head 保持 FP16，量化它精度损失明显
    )

    # 4. 执行量化（oneshot = 不训练，跑一次校准就出量化模型）
    print(f"\n开始量化（A100 上 7B 模型大约 20-40 分钟，比 GPTQ 快 2-3 倍）...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=n_samples,
    )

    # 5. 保存
    #    save_compressed=True 才会按压缩格式存（INT4 + 元数据），否则会反量化回 FP16 存
    print(f"\n保存到 {output_dir} ...")
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print("完成。可被 vLLM / SGLang / transformers 直接加载")


def compare_gptq_awq():
    """打印 GPTQ / AWQ / GGUF 对比表"""
    print("=" * 70)
    print("GPTQ vs AWQ vs GGUF 对比")
    print("=" * 70)
    print("""
| 特性          | GPTQ              | AWQ                | GGUF (llama.cpp)    |
|---------------|-------------------|--------------------|---------------------|
| 量化算法      | Hessian 逐层优化  | 激活感知权重保护   | 多种（Q4_K_M 等）   |
| 精度（一般）  | 好                | 更好               | 好                  |
| 量化速度      | 慢 (60-120 min)   | 较快 (20-40 min)   | 快（几分钟）        |
| 推理框架      | vLLM / SGLang / HF| vLLM / SGLang / HF | llama.cpp / Ollama  |
| GPU 推理速度  | 快                | 快                 | 较慢（非原生 CUDA） |
| CPU 推理      | 不支持            | 不支持             | 原生支持            |
| 校准数据      | 需要 (128~1024)   | 需要 (128~512)     | 不需要              |
| 主要用途      | GPU 部署          | GPU 部署           | 本地 / CPU / 边缘   |
""")


def main():
    parser = argparse.ArgumentParser(description="AWQ 模型量化（基于 llm-compressor）")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=str, default="./quantized-model-awq")
    parser.add_argument("--samples", type=int, default=512,
                        help="校准样本数（AWQ 比 GPTQ 需要的样本更少）")
    parser.add_argument("--compare", action="store_true", help="仅打印量化方法对比")
    args = parser.parse_args()

    if args.compare:
        compare_gptq_awq()
    else:
        quantize_awq(args.model, args.output, args.samples)


if __name__ == "__main__":
    main()
