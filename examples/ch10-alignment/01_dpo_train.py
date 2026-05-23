"""
用 TRL 做 DPO 训练

DPO（Direct Preference Optimization）直接从偏好数据优化模型，
不需要训练 Reward Model，不需要 PPO 的在线采样。

使用方法:
    python 01_dpo_train.py

    # 多卡训练
    torchrun --nproc_per_node=2 01_dpo_train.py

硬件要求:
    - 单卡 A100 40GB（bf16）
    - 或用 QLoRA + 单卡 24GB
"""

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # 用 Instruct 版本，已经过 SFT
OUTPUT_DIR = "./output/qwen2-7b-dpo"
DATA_PATH = "./data/sample_preference.jsonl"

# DPO 超参数
BETA = 0.1          # KL penalty 系数，越大对偏离 reference 惩罚越重
LEARNING_RATE = 1e-6 # DPO 通常用很小的学习率（比 SFT 小 10-100 倍）
NUM_EPOCHS = 1       # DPO 通常只跑 1 个 epoch
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8  # 有效 batch size = 2 * 8 = 16


# ============================================================
# 数据加载
# ============================================================
def load_preference_data(path: str) -> Dataset:
    """
    加载偏好数据。DPO 需要的格式：
    - prompt: 用户输入
    - chosen: 更好的回答
    - rejected: 较差的回答
    """
    import json
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


# ============================================================
# 主流程
# ============================================================
def main():
    print(f"Loading preference data from {DATA_PATH}")
    dataset = load_preference_data(DATA_PATH)
    print(f"Dataset size: {len(dataset)} preference pairs")

    # LoRA 配置（推荐在 DPO 中也用 LoRA，节省显存）
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # DPO 训练配置
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        loss_type="sigmoid",   # 标准 DPO loss（也支持 ipo, hinge 等变体）
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        max_length=1024,       # 最大序列长度
        max_prompt_length=512, # prompt 最大长度
    )

    print(f"Initializing DPO trainer with model: {MODEL_NAME}")
    # DPOTrainer 会自动：
    # 1. 加载模型作为 policy
    # 2. 创建 reference model（policy 的初始副本）
    # 3. 处理 chosen/rejected 的 tokenization
    trainer = DPOTrainer(
        model=MODEL_NAME,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    print("Key metrics to watch:")
    print("  - rewards/chosen: should increase")
    print("  - rewards/rejected: should decrease")
    print("  - rewards/margins: should increase (chosen - rejected)")
    print("  - rewards/accuracies: should be > 0.5 and increase")
    trainer.train()

    # 保存
    trainer.save_model(OUTPUT_DIR)
    print(f"DPO model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
