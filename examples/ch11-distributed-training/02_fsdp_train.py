"""
PyTorch FSDP（Fully Sharded Data Parallel）训练示例

FSDP 把模型参数、梯度、优化器状态分片到多张卡，
类似 DeepSpeed ZeRO Stage 3，但是 PyTorch 原生支持。

使用方法:
    torchrun --nproc_per_node=4 02_fsdp_train.py

硬件要求:
    - 多卡 GPU（至少 2 张）
    - 推荐 NVLink 互联（FSDP 通信量较大）
"""

import os
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)


# ============================================================
# 模型定义（和 DDP 示例相同）
# ============================================================
class TransformerBlock(nn.Module):
    """单个 Transformer 层，FSDP 会以此为单位做分片"""
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm architecture
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, num_layers=12, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits


class DummyDataset(Dataset):
    def __init__(self, size=10000, seq_len=128, vocab_size=32000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


# ============================================================
# FSDP 训练
# ============================================================
def train():
    # 初始化分布式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    is_main = (global_rank == 0)

    if is_main:
        print(f"Starting FSDP training with {world_size} GPUs")

    # 1. 创建模型（先在 CPU 上创建，FSDP 会自动分发）
    model = SimpleTransformer()

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params / 1e6:.1f}M")

    # 2. 配置混合精度
    # FSDP 的混合精度比 DDP 更细粒度：
    # - param_dtype: 参数存储精度
    # - reduce_dtype: 梯度 reduce 精度
    # - buffer_dtype: buffer 精度
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # 3. 配置自动包装策略
    # FSDP 需要知道以哪个模块为单位做分片
    # transformer_auto_wrap_policy 会对指定的模块类做包装
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},  # 以 TransformerBlock 为单位分片
    )

    # 4. 用 FSDP 包装模型
    model = FSDP(
        model,
        # ShardingStrategy 选择:
        # FULL_SHARD: 类似 ZeRO-3，参数+梯度+优化器都分片（最省显存）
        # SHARD_GRAD_OP: 类似 ZeRO-2，只分片梯度和优化器
        # NO_SHARD: 等同于 DDP
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        # CPU offload（取消注释可启用，进一步降低显存但训练变慢）
        # cpu_offload=CPUOffload(offload_params=True),
    )

    if is_main:
        print(f"FSDP model created with FULL_SHARD strategy")
        # FSDP 下参数被分片了，每张卡只持有 1/N 的参数
        local_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters per GPU: {local_params / 1e6:.1f}M "
              f"(full model / {world_size})")

    # 5. 优化器（必须在 FSDP 包装之后创��）
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 6. 数据加载
    dataset = DummyDataset(size=10000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank)
    dataloader = DataLoader(
        dataset, batch_size=8, sampler=sampler, num_workers=2, pin_memory=True,
    )

    # 7. 训练循环
    num_epochs = 3
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(local_rank)
            labels = batch["labels"].to(local_rank)

            loss, _ = model(input_ids, labels)
            loss.backward()

            # FSDP 下的梯度裁剪需要用 model.clip_grad_norm_
            model.clip_grad_norm_(max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_steps += 1

            if is_main and step % 50 == 0:
                print(f"  Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_steps
        if is_main:
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # 8. 保存模型
    # FSDP 的模型保存比 DDP 复杂，需要用 FSDP 的 state_dict API
    if is_main:
        print("Saving model...")

    # 方式一：用 FULL_STATE_DICT（收集到一张卡上保存，简单但内存开销大）
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        state_dict = model.state_dict()
        if is_main:
            torch.save(state_dict, "fsdp_model.pt")
            print("Model saved to fsdp_model.pt")

    # 方式二：用 SHARDED_STATE_DICT（推荐用于大模型）
    # from torch.distributed.fsdp import ShardedStateDictConfig
    # sharded_config = ShardedStateDictConfig(offload_to_cpu=True)
    # with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_config):
    #     state_dict = model.state_dict()
    #     # 保存分片 checkpoint（每张卡保存自己的部分）
    #     torch.save(state_dict, f"fsdp_model_shard_{global_rank}.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
