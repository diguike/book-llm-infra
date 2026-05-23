#!/usr/bin/env python3
"""
简化版 nanoGPT：字符级 GPT 模型训练。

基于 Andrej Karpathy 的 nanoGPT 简化，去掉 BPE tokenizer，
直接用字符作为 token，专注于理解 Transformer 架构本身。

默认数据集：tinyShakespeare（首次运行自动下载，~1.1 MB）
离线兜底：内置中文短文本（仅供脚本能跑通，效果极差）

用法:
    # 默认用 tinyShakespeare 训练
    python 04_nano_gpt_train.py --train --epochs 20

    # 用自定义文本文件训练
    python 04_nano_gpt_train.py --train --data-file my_text.txt --epochs 20

    # 训练后生成（prompt 默认 "ROMEO:" 配合 Shakespeare 数据集）
    python 04_nano_gpt_train.py --generate --prompt "ROMEO:"

    # 一键训练+生成
    python 04_nano_gpt_train.py --train --generate --epochs 20

依赖:
    pip install torch
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("缺少依赖: pip install torch")
    sys.exit(1)


# ─────────────────────────────────────────────
# 模型定义
# ─────────────────────────────────────────────


class SelfAttention(nn.Module):
    """单头自注意力。"""

    def __init__(self, n_embd: int, head_dim: int, block_size: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(n_embd, head_dim, bias=False)
        self.key = nn.Linear(n_embd, head_dim, bias=False)
        self.value = nn.Linear(n_embd, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Causal mask: 下三角矩阵，确保 token 只能看到自己和之前的 token
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.query(x)   # [B, T, head_dim]
        k = self.key(x)     # [B, T, head_dim]
        v = self.value(x)   # [B, T, head_dim]

        # Attention scores: Q @ K^T / sqrt(d)
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # [B, T, T]
        # 应用 causal mask
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # [B, T, T]
        weights = self.dropout(weights)

        out = weights @ v  # [B, T, head_dim]
        return out


class MultiHeadAttention(nn.Module):
    """多头注意力 = 多个单头并行 + 线性投影。"""

    def __init__(
        self, n_embd: int, n_head: int, block_size: int, dropout: float
    ):
        super().__init__()
        assert n_embd % n_head == 0
        head_dim = n_embd // n_head
        self.heads = nn.ModuleList(
            [SelfAttention(n_embd, head_dim, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 所有头并行计算，结果拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # [B, T, n_embd]
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """前馈网络: Linear → GELU → Linear。"""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """一个 Transformer Block = LayerNorm → Attention → Residual → LayerNorm → FFN → Residual。"""

    def __init__(
        self, n_embd: int, n_head: int, block_size: int, dropout: float
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # Pre-norm + Residual
        x = x + self.ffn(self.ln2(x))    # Pre-norm + Residual
        return x


class MiniGPT(nn.Module):
    """
    最小 GPT 模型（字符级）。

    结构:
        Token Embedding + Position Embedding
        → N × TransformerBlock
        → LayerNorm → Linear Head → Logits
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # 参数初始化
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"模型参数量: {n_params / 1e3:.1f}K")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度 {T} 超过 block_size {self.block_size}"

        # Token + Position Embedding
        tok_emb = self.token_embedding(idx)  # [B, T, n_embd]
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )  # [T, n_embd]
        x = tok_emb + pos_emb  # [B, T, n_embd] + [T, n_embd] → broadcast

        # Transformer Blocks
        x = self.blocks(x)    # [B, T, n_embd]
        x = self.ln_f(x)      # [B, T, n_embd]

        # Language Model Head
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # 计算 loss（如果提供了 targets）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """自回归生成。"""
        for _ in range(max_new_tokens):
            # 截断到 block_size
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # [B, vocab_size]

            # Top-k 采样
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=-1)

        return idx


# ─────────────────────────────────────────────
# 数据处理
# ─────────────────────────────────────────────

# tinyShakespeare 数据集：Karpathy 在 char-rnn / nanoGPT 里用的经典字符级语料
# ~1.1 MB 莎士比亚戏剧对白，全 ASCII，字符词表只有 65 个，是验证字符级 GPT 最常用的数据集
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinyshakespeare.txt")

# 离线兜底数据：仅供没网时脚本能跑通流程，~700 字根本训不出什么东西
BUILTIN_TEXT = """
大语言模型是一种基于 Transformer 架构的深度学习模型。它通过在大量文本数据上进行预训练，学习到语言的统计规律和知识。
训练过程分为两个阶段：预训练和微调。预训练阶段使用大规模无标注文本，通过下一个词预测任务来学习。微调阶段在特定任务的标注数据上进行调整。
推理优化是部署大语言模型的关键。常见的优化技术包括：量化、KV Cache、FlashAttention、连续批处理等。这些技术可以显著降低延迟和成本。
GPU 是运行大语言模型的核心硬件。NVIDIA 的 A100 和 H100 是目前最常用的训练和推理 GPU。显存大小决定了能运行的模型规模。
Transformer 的核心是自注意力机制。每个 token 都会关注序列中的其他 token，计算注意力权重。这种机制使模型能够捕获长距离依赖关系。
分布式训练和推理技术使得超大规模模型成为可能。张量并行、流水线并行和数据并行是三种主要的并行策略。
开源大模型社区蓬勃发展。Llama、Qwen、Mistral 等开源模型的能力不断提升，推动了整个行业的进步。
Agent 是大语言模型的重要应用形式。通过工具调用和多步推理，Agent 可以完成复杂的任务。
从前端到后端，从应用到基础设施，大语言模型正在改变软件开发的方方面面。
理解底层原理对于构建高质量的 AI 应用至关重要。不了解推理引擎的工作方式，就无法做出正确的架构决策。
""".strip()


def load_shakespeare() -> str | None:
    """下载 tinyShakespeare 到本地缓存。已存在则直接读，下载失败返回 None。"""
    if os.path.exists(SHAKESPEARE_CACHE):
        with open(SHAKESPEARE_CACHE, "r", encoding="utf-8") as f:
            return f.read()

    print(f"首次运行，正在下载 tinyShakespeare 数据集 (~1.1 MB) ...")
    print(f"  来源: {SHAKESPEARE_URL}")
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_CACHE)
        with open(SHAKESPEARE_CACHE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"  下载失败: {e}")
        print(f"  回退到内置中文短文本（数据量太小，仅用于流程验证）")
        return None


class CharDataset:
    """字符级数据集。"""

    def __init__(self, text: str, block_size: int):
        # 构建字符词表
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size

        # 编码整个文本
        self.data = torch.tensor(
            [self.stoi[ch] for ch in text], dtype=torch.long
        )

        print(f"数据集: {len(text)} 个字符, 词表大小: {self.vocab_size}")
        print(f"前 50 个字符: {text[:50]}...")

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(
            [self.stoi.get(ch, 0) for ch in text], dtype=torch.long
        )

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos.get(i, "?") for i in ids)

    def get_batch(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """随机采样一个 batch。"""
        max_start = len(self.data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"文本太短 ({len(self.data)} chars)，需要至少 {self.block_size + 1} chars"
            )

        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s : s + self.block_size] for s in starts])
        y = torch.stack(
            [self.data[s + 1 : s + self.block_size + 1] for s in starts]
        )
        return x.to(device), y.to(device)


# ─────────────────────────────────────────────
# 训练和生成
# ─────────────────────────────────────────────


def train(args):
    """训练模型。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据：优先 --data-file，其次 tinyShakespeare，下载失败退化到内置中文文本
    if args.data_file and os.path.exists(args.data_file):
        with open(args.data_file, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"从文件加载数据: {args.data_file}")
    else:
        text = load_shakespeare()
        if text is None:
            text = BUILTIN_TEXT
            print("使用内置中文短文本")
        else:
            print(f"使用 tinyShakespeare 数据集（{len(text)} 字符）")

    dataset = CharDataset(text, block_size=args.block_size)

    # 创建模型
    model = MiniGPT(
        vocab_size=dataset.vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=args.dropout,
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 学习率调度：cosine decay with warmup
    warmup_steps = args.epochs * 50 // 10  # 10% warmup
    total_steps = args.epochs * 50

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练循环
    print(f"\n开始训练: {args.epochs} epochs, batch_size={args.batch_size}")
    print("-" * 50)

    steps_per_epoch = 50  # 每个 epoch 的步数
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step in range(steps_per_epoch):
            xb, yb = dataset.get_batch(args.batch_size, device)
            logits, loss = model(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch
        elapsed = time.time() - start_time
        lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:4d}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # 生成样本
            model.eval()
            sample_prompt = text[:10]
            sample_ids = dataset.encode(sample_prompt).unsqueeze(0).to(device)
            generated = model.generate(sample_ids, max_new_tokens=50)
            sample_text = dataset.decode(generated[0])
            print(f"  Sample: {sample_text[:80]}...")

    print("-" * 50)
    total_time = time.time() - start_time
    print(f"训练完成! 总耗时: {total_time:.1f}s")

    # 保存模型和词表
    save_path = args.save_path
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": {"stoi": dataset.stoi, "itos": dataset.itos},
            "config": {
                "vocab_size": dataset.vocab_size,
                "n_embd": args.n_embd,
                "n_head": args.n_head,
                "n_layer": args.n_layer,
                "block_size": args.block_size,
            },
        },
        save_path,
    )
    print(f"模型已保存: {save_path}")

    return model, dataset


def generate(args):
    """加载模型并生成文本。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.save_path):
        print(f"模型文件不存在: {args.save_path}")
        print("请先运行: python 04_nano_gpt_train.py --train")
        sys.exit(1)

    # 加载
    checkpoint = torch.load(args.save_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    stoi = vocab["stoi"]
    itos = vocab["itos"]

    model = MiniGPT(
        vocab_size=config["vocab_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        block_size=config["block_size"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 编码 prompt
    prompt = args.prompt
    ids = torch.tensor(
        [stoi.get(ch, 0) for ch in prompt], dtype=torch.long
    ).unsqueeze(0).to(device)

    # 生成
    print(f"\nPrompt: {prompt}")
    print(f"生成 {args.max_tokens} 个 token ...\n")

    start = time.perf_counter()
    generated = model.generate(
        ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    elapsed = time.perf_counter() - start

    # 解码
    text = "".join(itos.get(i, "?") for i in generated[0].tolist())
    print(f"生成结果:\n{text}")
    print(f"\n耗时: {elapsed * 1000:.1f}ms, {args.max_tokens / elapsed:.1f} tokens/s")


def main():
    parser = argparse.ArgumentParser(
        description="最小 GPT 字符级训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 模式
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--generate", action="store_true", help="生成文本")

    # 数据
    parser.add_argument(
        "--data-file", type=str, default=None, help="训练数据文件路径（纯文本）"
    )

    # 模型配置
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding 维度 (default: 128)")
    parser.add_argument("--n-head", type=int, default=4, help="注意力头数 (default: 4)")
    parser.add_argument("--n-layer", type=int, default=4, help="Transformer 层数 (default: 4)")
    parser.add_argument("--block-size", type=int, default=128, help="最大序列长度 (default: 128)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 率 (default: 0.1)")

    # 训练配置
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数 (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率 (default: 3e-4)")

    # 生成配置
    parser.add_argument("--prompt", type=str, default="ROMEO:", help="生成用的 prompt（默认配合 Shakespeare 数据集）")
    parser.add_argument("--max-tokens", type=int, default=200, help="生成 token 数 (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度 (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k 采样 (default: 50)")

    # 保存
    parser.add_argument(
        "--save-path",
        type=str,
        default="mini_gpt_char.pt",
        help="模型保存路径 (default: mini_gpt_char.pt)",
    )

    args = parser.parse_args()

    if not args.train and not args.generate:
        parser.print_help()
        print("\n请指定 --train 或 --generate（或两者都指定）")
        sys.exit(1)

    if args.train:
        model, dataset = train(args)

    if args.generate:
        generate(args)


if __name__ == "__main__":
    main()
