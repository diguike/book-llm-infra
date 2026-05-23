"""
训练一个简单的 Reward Model

Reward Model 接收 (prompt, response) 对，输出一个标量分数。
在 RLHF 的 PPO 阶段用来指导模型优化。

虽然 DPO 不需要显式的 Reward Model，但理解 RM 的原理对理解 RLHF 很有帮助。
而且训练好的 RM 可以用于数据筛选、质量评估等场景。

使用方法:
    python 02_reward_model.py

硬件要求:
    - 单卡 A10 24GB（配合 LoRA + 量化）
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
DATA_PATH = "./data/sample_preference.jsonl"
OUTPUT_DIR = "./output/qwen2-7b-reward-model"

LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8


# ============================================================
# 数据处理
# ============================================================
def load_preference_data(path: str) -> Dataset:
    """
    加载偏好数据。RewardTrainer 需要的格式和 DPO 一样：
    - prompt (可选): 用户输入
    - chosen: 更好的回答
    - rejected: 较差的回答

    RewardTrainer 会自动把 chosen 和 rejected 分别和 prompt 拼接，
    然后训练模型让 chosen 的分数高于 rejected。
    """
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
    print("Loading preference data...")
    dataset = load_preference_data(DATA_PATH)
    print(f"Dataset size: {len(dataset)} preference pairs")

    # LoRA 配置
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        task_type="SEQ_CLS",  # Reward Model 是序列分类任务
    )

    # 训练配置
    training_args = RewardConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        max_length=1024,
    )

    print(f"Initializing RewardTrainer with model: {MODEL_NAME}")
    # RewardTrainer 会自动：
    # 1. 在模型输出层上加一个 score head（线性层，输出标量）
    # 2. 用 Bradley-Terry 排序损失训练
    #    loss = -log(sigmoid(score_chosen - score_rejected))
    trainer = RewardTrainer(
        model=MODEL_NAME,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting Reward Model training...")
    print("Key metric: accuracy (chosen score > rejected score)")
    trainer.train()

    # 保存
    trainer.save_model(OUTPUT_DIR)
    print(f"Reward Model saved to {OUTPUT_DIR}")

    # RM 的使用方式示例
    print("\nReward Model usage example:")
    print('  from transformers import pipeline')
    print(f'  rm = pipeline("text-classification", model="{OUTPUT_DIR}")')
    print('  score = rm("Your prompt + response text")')


if __name__ == "__main__":
    main()
