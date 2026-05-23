"""
QLoRA 微调完整流程

在 4-bit 量化的基础模型上做 LoRA 微调，大幅降低显存需求。
7B 模型只需要 ~7GB 显存，在 RTX 3090/4090 或 A10 上轻松运行。

使用方法:
    python 02_qlora_train.py

硬件要求:
    - 单卡 24GB GPU（A10、RTX 4090、RTX 3090）
    - 甚至 16GB GPU 也可能跑得动（减小 batch_size）
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2-7B"
DATA_PATH = "./data/sample_train.jsonl"
OUTPUT_DIR = "./output/qwen2-7b-qlora"
MAX_SEQ_LEN = 1024

# QLoRA 量化配置
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,              # 4-bit 量化
    bnb_4bit_quant_type="nf4",      # NormalFloat4，比普通 int4 更好
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时用 bf16
    bnb_4bit_use_double_quant=True,  # 双重量化，进一步节省显存
)

# LoRA 配置（和 LoRA 完全一样）
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 训练超参数
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4


# ============================================================
# 数据处理（和 01_lora_from_scratch.py 相同）
# ============================================================
def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def preprocess_function(examples, tokenizer, max_length=MAX_SEQ_LEN):
    input_ids_list = []
    labels_list = []

    for messages in examples["messages"]:
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
        labels = input_ids.copy()
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "labels": labels_list}


# ============================================================
# 主流程
# ============================================================
def main():
    print(f"Loading model with 4-bit quantization: {MODEL_NAME}")

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载量化模型
    # 这里模型会以 4-bit 加载，显存占用从 15GB 降到 ~4GB
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. 准备量化模型做训练
    # 这一步会：冻结量化参数、设置 gradient checkpointing 等
    model = prepare_model_for_kbit_training(model)

    # 4. 配置 LoRA（和非量化版本完全一样）
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 加载数据
    raw_data = load_jsonl(DATA_PATH)
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    print(f"Dataset size: {len(dataset)} samples")

    # 6. 训练参数
    # 注意：QLoRA 不支持 bf16=True（因为模型本身是量化的）
    # 但 compute_dtype 是 bf16，所以计算仍然用 bf16
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
        optim="paged_adamw_8bit",  # 8-bit Adam，进一步节省显存
        report_to="none",
        dataloader_num_workers=4,
        max_grad_norm=0.3,  # QLoRA 论文推荐的梯度裁剪值
    )

    # 7. 训练
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

    print("Starting QLoRA training...")
    print(f"Expected GPU memory usage: ~6-7 GB")
    trainer.train()

    # 8. 保存
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"QLoRA adapter saved to {OUTPUT_DIR}")
    print(f"Next step: use 03_merge_lora.py to merge adapter into base model")


if __name__ == "__main__":
    main()
