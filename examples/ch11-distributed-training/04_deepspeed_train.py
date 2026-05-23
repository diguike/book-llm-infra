"""
DeepSpeed 训练示例

演示两种使用 DeepSpeed 的方式：
1. 和 HuggingFace Transformers 集成（推荐，最简单）
2. 原生 DeepSpeed API（更灵活）

使用方法:
    # 方式一：通过 HuggingFace Trainer（推荐）
    deepspeed --num_gpus=4 04_deepspeed_train.py --mode huggingface

    # 方式二：原生 DeepSpeed API
    deepspeed --num_gpus=4 04_deepspeed_train.py --mode native

    # 多机训练
    deepspeed --num_gpus=4 --num_nodes=2 --hostfile hostfile \
        04_deepspeed_train.py --mode huggingface
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ============================================================
# 共用的数据集和模型
# ============================================================
class DummyDataset(Dataset):
    """模拟训练数据"""
    def __init__(self, size=10000, seq_len=512, vocab_size=32000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


# ============================================================
# 方式一：HuggingFace Transformers + DeepSpeed（推荐）
# ============================================================
def train_with_huggingface():
    """
    最简单的方式：在 TrainingArguments 里指定 deepspeed 配置文件。
    HuggingFace Trainer 会自动处理 DeepSpeed 的初始化、训练、保存。
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )

    model_name = "Qwen/Qwen2-7B"  # 实际使用时换成你的模型
    ds_config = os.path.join(os.path.dirname(__file__), "03_deepspeed_config.json")

    print(f"Loading model: {model_name}")
    print(f"DeepSpeed config: {ds_config}")

    # 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 注意：使用 DeepSpeed 时，不要设置 device_map="auto"
    # DeepSpeed 会自己管理设备分配
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 创建数据集（实际项目中替换为你的数据）
    dataset = DummyDataset(size=1000)

    # 训练参数——关键是 deepspeed 参数
    training_args = TrainingArguments(
        output_dir="./output/deepspeed-demo",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to="none",
        # 关键：指定 DeepSpeed 配置文件
        deepspeed=ds_config,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training with DeepSpeed ZeRO-2...")
    trainer.train()
    trainer.save_model()
    print("Training complete!")


# ============================================================
# 方式二：原生 DeepSpeed API
# ============================================================
class SimpleModel(nn.Module):
    """用于演示原生 API 的简单模型"""
    def __init__(self, vocab_size=32000, d_model=512, num_layers=6, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits


def train_with_native_api():
    """
    使用 DeepSpeed 原生 API。
    更灵活，但代码也更多。适合需要精细控制训练流程的场景。
    """
    import deepspeed
    from torch.utils.data import DataLoader, DistributedSampler

    # DeepSpeed 配置（也可以从 JSON 文件加载）
    ds_config = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            }
        },
        "scheduler": {
            "type": "WarmupCosineWithMinLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 100,
                "total_num_steps": 1000,
            }
        },
    }

    # 创建模型
    model = SimpleModel()
    total_params = sum(p.numel() for p in model.parameters())

    # DeepSpeed 初始化
    # 这一步会：
    # 1. 初始化分布式环境
    # 2. 包装模型
    # 3. 创建优化器和 scheduler
    # 4. 设置 ZeRO 优化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    local_rank = model_engine.local_rank
    is_main = (local_rank == 0)

    if is_main:
        print(f"Model parameters: {total_params / 1e6:.1f}M")
        print(f"Training with DeepSpeed ZeRO Stage 2")

    # 数据加载
    dataset = DummyDataset(size=10000, seq_len=128)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=ds_config["train_micro_batch_size_per_gpu"],
        sampler=sampler, num_workers=2,
    )

    # 训练循环
    num_epochs = 3
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(local_rank)
            labels = batch["labels"].to(local_rank)

            # DeepSpeed engine 封装了 forward + backward + step
            loss, _ = model_engine(input_ids, labels)
            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            num_steps += 1

            if is_main and step % 50 == 0:
                print(f"  Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_steps
        if is_main:
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # 保存 checkpoint
    if is_main:
        model_engine.save_checkpoint("./output/deepspeed-native", tag="final")
        print("Model checkpoint saved!")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["huggingface", "native"],
        default="huggingface",
        help="Training mode: huggingface (recommended) or native DeepSpeed API"
    )
    # DeepSpeed 会添加自己的命令行参数，需要用 deepspeed 的 parser
    parser.add_argument("--local_rank", type=int, default=-1)
    args, _ = parser.parse_known_args()

    if args.mode == "huggingface":
        train_with_huggingface()
    else:
        train_with_native_api()
