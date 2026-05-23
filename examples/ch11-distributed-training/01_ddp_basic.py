"""
PyTorch DDP（DistributedDataParallel）基础示例

DDP 是最简单的分布式训练策略：每张卡一份模型，数据不同，梯度同步。

使用方法:
    # 单机 4 卡
    torchrun --nproc_per_node=4 01_ddp_basic.py

    # 多机（机器 0）
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
        --master_addr=192.168.1.1 --master_port=29500 01_ddp_basic.py

    # 多机（机器 1）
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
        --master_addr=192.168.1.1 --master_port=29500 01_ddp_basic.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# ============================================================
# 模拟数据集
# ============================================================
class DummyDataset(Dataset):
    """简单的示例数据集"""
    def __init__(self, size=1000, seq_len=128, vocab_size=32000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 随机生成 input_ids 和 labels
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


# ============================================================
# 简单的 Transformer 模型
# ============================================================
class SimpleTransformer(nn.Module):
    """一个简化的 Transformer 模型，用于演示 DDP"""
    def __init__(self, vocab_size=32000, d_model=512, nhead=8, num_layers=6):
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


# ============================================================
# DDP 训练
# ============================================================
def setup_distributed():
    """初始化分布式环境"""
    # torchrun 会自动设置这些环境变量
    dist.init_process_group(backend="nccl")  # NVIDIA GPU 用 nccl

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    return local_rank, global_rank, world_size


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def train():
    local_rank, global_rank, world_size = setup_distributed()
    is_main = (global_rank == 0)

    if is_main:
        print(f"Starting DDP training with {world_size} GPUs")

    # 1. 创建模型并放到对应 GPU
    model = SimpleTransformer().to(local_rank)

    # 2. 用 DDP 包装模型
    # DDP 会在每个 backward 后自动做 all-reduce 同步梯���
    model = DDP(model, device_ids=[local_rank])

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params / 1e6:.1f}M")

    # 3. 创建数据集和 DistributedSampler
    dataset = DummyDataset(size=10000)

    # DistributedSampler 确保每张卡拿到不同的数据子集
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 5. 训练循环
    num_epochs = 3
    for epoch in range(num_epochs):
        # 重要：每个 epoch 设置 sampler 的 epoch，确保数据 shuffle 不同
        sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(dataloader):
            # 数据移到对应 GPU
            input_ids = batch["input_ids"].to(local_rank)
            labels = batch["labels"].to(local_rank)

            # Forward
            loss, _ = model(input_ids, labels)

            # Backward（DDP 会自动做梯度 all-reduce）
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参���
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_steps += 1

            if is_main and step % 50 == 0:
                print(f"  Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        # 打印 epoch 汇总（只在主进程打印）
        avg_loss = total_loss / num_steps
        if is_main:
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # 6. 保存模型（只在主进程保存）
    if is_main:
        # DDP 包装的模型，实际模型在 model.module 里
        torch.save(model.module.state_dict(), "ddp_model.pt")
        print("Model saved to ddp_model.pt")

    cleanup()


if __name__ == "__main__":
    train()
