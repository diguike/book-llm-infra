"""
用 PEFT 库从零做 LoRA 微调

演示如何用 HuggingFace Transformers + PEFT 对 Qwen2-7B 进行 LoRA 微调。
这是最基础的方式，适合想理解每一步细节的场景。

使用方法:
    python 01_lora_from_scratch.py

硬件要求:
    - 单卡 A100 40GB 或 A10 24GB（bf16 LoRA）
    - 如果显存不够，请使用 02_qlora_train.py（4-bit 量化）
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2-7B"          # 基础模型
DATA_PATH = "./data/sample_train.jsonl" # 训练数据
OUTPUT_DIR = "./output/qwen2-7b-lora"   # 输出目录
MAX_SEQ_LEN = 1024                      # 最大序列长度

# LoRA 超参数
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 训练超参数
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # 有效 batch size = 4 * 4 = 16


# ============================================================
# 数据处理
# ============================================================
def load_jsonl(path: str) -> list[dict]:
    """加载 JSONL 格式的数据文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def preprocess_function(examples, tokenizer, max_length=MAX_SEQ_LEN):
    """
    把 messages 格式的数据转成 input_ids + labels。
    labels 中 user 部分设为 -100（不计算 loss），只在 assistant 回答上计算 loss。
    """
    input_ids_list = []
    labels_list = []

    for messages in examples["messages"]:
        # 用 tokenizer 的 chat template 拼接对话
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        input_ids = tokenized["input_ids"]

        # 简化处理：对整个序列计算 loss
        # 更精细的做法是只对 assistant 回复部分计算 loss
        labels = input_ids.copy()

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "labels": labels_list}


# ============================================================
# 主流程
# ============================================================
def main():
    print(f"Loading model: {MODEL_NAME}")

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型（bf16 精度）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()  # LoRA 需要这一步

    # 3. 配置 LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",  # 对所有线性层加 LoRA
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # 打印可训练参数信息
    model.print_trainable_parameters()
    # 预期输出类似: trainable params: 83,886,080 || all params: 7,699,898,368 || trainable%: 1.09

    # 4. 加载和处理数据
    raw_data = load_jsonl(DATA_PATH)
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    print(f"Dataset size: {len(dataset)} samples")

    # 5. 配置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",  # 不上报到 wandb 等平台
        dataloader_num_workers=4,
    )

    # 6. 创建 Trainer 并开始训练
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # 7. 保存 LoRA adapter（只保存新增参数，通常几十 MB）
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
