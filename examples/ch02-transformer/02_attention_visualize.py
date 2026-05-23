#!/usr/bin/env python3
"""
可视化 Transformer 模型的 Attention 权重。

加载一个预训练模型（默认 GPT-2），对输入文本运行前向传播，
提取并可视化指定层和头的 Attention 权重热力图。

用法:
    python 02_attention_visualize.py
    python 02_attention_visualize.py --model gpt2 --text "The cat sat on the mat"
    python 02_attention_visualize.py --layer 5 --head 0 --output attention.png

依赖:
    pip install torch transformers matplotlib numpy
"""

import argparse
import sys

import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import matplotlib

    matplotlib.use("Agg")  # 无头环境下使用非交互后端
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("安装: pip install torch transformers matplotlib numpy")
    sys.exit(1)


def get_attention_weights(
    model_name: str, text: str
) -> tuple[list[str], torch.Tensor]:
    """
    加载模型并提取 attention 权重。

    Returns:
        tokens: token 文本列表
        attentions: shape [n_layers, n_heads, seq_len, seq_len]
    """
    print(f"加载模型: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, torch_dtype=torch.float32
    )
    model.eval()

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([t]) for t in token_ids]

    print(f"Token 数: {len(tokens)}")
    print(f"Tokens: {tokens}")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions: tuple of (batch, n_heads, seq_len, seq_len)，每层一个
    attentions = torch.stack(
        [layer_attn[0] for layer_attn in outputs.attentions]
    )  # [n_layers, n_heads, seq_len, seq_len]

    n_layers, n_heads, seq_len, _ = attentions.shape
    print(f"模型层数: {n_layers}, 注意力头数: {n_heads}")

    return tokens, attentions


def plot_attention_heatmap(
    tokens: list[str],
    attention: np.ndarray,
    layer: int,
    head: int,
    output_path: str,
):
    """绘制单个 attention head 的热力图。"""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(attention, cmap="Blues", aspect="auto")

    # 设置坐标轴标签
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)

    ax.set_xlabel("Key (被关注的 token)")
    ax.set_ylabel("Query (正在处理的 token)")
    ax.set_title(f"Attention Weights - Layer {layer}, Head {head}")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"热力图已保存: {output_path}")
    plt.close()


def plot_attention_multi_head(
    tokens: list[str],
    attentions: np.ndarray,
    layer: int,
    n_heads_to_show: int,
    output_path: str,
):
    """绘制一层中多个 head 的 attention 对比图。"""
    n_heads = attentions.shape[0]
    n_show = min(n_heads_to_show, n_heads)

    cols = min(4, n_show)
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(n_show):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.imshow(attentions[i], cmap="Blues", aspect="auto")
        ax.set_title(f"Head {i}", fontsize=10)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)

    # 隐藏多余的子图
    for i in range(n_show, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"Layer {layer} - Multiple Attention Heads", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"多头对比图已保存: {output_path}")
    plt.close()


def print_attention_stats(tokens: list[str], attentions: torch.Tensor):
    """打印 attention 的统计信息。"""
    n_layers, n_heads, seq_len, _ = attentions.shape

    print("\n" + "=" * 60)
    print("Attention 统计")
    print("=" * 60)

    # 分析最后一层：每个头关注的模式
    last_layer = attentions[-1].numpy()  # [n_heads, seq_len, seq_len]

    print(f"\n最后一层 (Layer {n_layers - 1}) 的 Attention 模式:")
    for h in range(min(4, n_heads)):
        # 每个 query token 最关注哪个 key token
        max_attn_idx = last_layer[h].argmax(axis=-1)
        avg_attn_entropy = -(
            last_layer[h] * np.log(last_layer[h] + 1e-10)
        ).sum(axis=-1).mean()
        print(f"  Head {h}: 平均熵={avg_attn_entropy:.3f}", end="")
        if avg_attn_entropy < 0.5:
            print(" (集中型: 关注少数 token)")
        elif avg_attn_entropy > 2.0:
            print(" (分散型: 均匀关注)")
        else:
            print(" (中等)")

    # 分析所有层的平均 attention 距离
    print(f"\n各层平均 attention 距离（值越大表示关注越远的 token）:")
    for layer_idx in range(0, n_layers, max(1, n_layers // 8)):
        layer_attn = attentions[layer_idx].numpy()  # [n_heads, seq_len, seq_len]
        # 计算加权平均距离
        positions = np.arange(seq_len)
        distances = np.abs(positions[:, None] - positions[None, :])  # [seq_len, seq_len]
        avg_dist = (layer_attn * distances[None, :, :]).sum(axis=-1).mean()
        print(f"  Layer {layer_idx:2d}: avg_distance = {avg_dist:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="可视化 Transformer 的 Attention 权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace 模型名 (default: gpt2)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The cat sat on the mat and looked at the dog",
        help="输入文本",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="可视化的层编号，-1 表示最后一层 (default: -1)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="可视化的注意力头编号 (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="attention_heatmap.png",
        help="输出图片路径 (default: attention_heatmap.png)",
    )
    parser.add_argument(
        "--multi-head",
        action="store_true",
        help="绘制多个 head 的对比图",
    )
    parser.add_argument(
        "--n-heads-show",
        type=int,
        default=8,
        help="多头模式下显示的头数 (default: 8)",
    )
    args = parser.parse_args()

    # 获取 attention 权重
    tokens, attentions = get_attention_weights(args.model, args.text)

    n_layers = attentions.shape[0]
    layer_idx = args.layer if args.layer >= 0 else n_layers + args.layer

    if layer_idx < 0 or layer_idx >= n_layers:
        print(f"错误: layer 必须在 0-{n_layers - 1} 之间，或 -1 表示最后一层")
        sys.exit(1)

    # 打印统计信息
    print_attention_stats(tokens, attentions)

    # 绘制热力图
    if args.multi_head:
        layer_attentions = attentions[layer_idx].numpy()
        multi_output = args.output.replace(".png", "_multi.png")
        plot_attention_multi_head(
            tokens, layer_attentions, layer_idx, args.n_heads_show, multi_output
        )
    else:
        attention = attentions[layer_idx, args.head].numpy()
        plot_attention_heatmap(tokens, attention, layer_idx, args.head, args.output)

    print("\n提示: 观察不同 head 的 pattern 差异——")
    print("  有的 head 关注前一个 token (local)，有的关注句首 (global)，")
    print("  有的关注特定语法关系 (如动词-宾语)。")
    print("  这种多样性是 Multi-Head Attention 的核心价值。")


if __name__ == "__main__":
    main()
