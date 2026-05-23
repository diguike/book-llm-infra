"""
Embedding 模型速度与效果对比基准测试

对比不同 Embedding 模型的：
  - 编码速度（单条 / 批量）
  - 向量维度
  - 中文语义相似度效果

运行：
  pip install -r requirements.txt
  python 01_embedding_benchmark.py

注意：首次运行会下载模型权重，需要足够的磁盘空间和网络。
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding_benchmark")


# ============================================================
# 测试数据
# ============================================================

# 语义相似度测试对（中文）
SIMILARITY_PAIRS = [
    # (句子 A, 句子 B, 预期相似度: high/medium/low)
    ("机器学习是人工智能的一个分支", "ML 是 AI 的子领域", "high"),
    ("今天天气真好", "今天阳光明媚", "high"),
    ("Python 是一种编程语言", "蟒蛇是一种蛇类", "low"),
    ("深度学习需要大量数据", "深度学习对数据量要求很高", "high"),
    ("我喜欢吃苹果", "苹果公司发布了新产品", "low"),
    ("向量数据库用于存储嵌入向量", "矢量数据库适合保存 embedding", "high"),
    ("K8s 是容器编排平台", "Kubernetes 管理容器化应用", "high"),
    ("今天股市大跌", "A 股今日暴跌", "high"),
    ("猫在沙发上睡觉", "数据库查询优化", "low"),
    ("RAG 系统结合了检索和生成", "检索增强生成技术", "high"),
]

# 批量编码性能测试文本
BATCH_TEXTS = [
    f"这是第 {i} 条测试文本，用于测试 Embedding 模型的批量编码性能。"
    f"文本长度需要有一定变化，这样测试结果更有参考价值。" + "填充内容。" * (i % 5)
    for i in range(100)
]


# ============================================================
# 模型配置
# ============================================================

@dataclass
class ModelConfig:
    name: str
    model_id: str
    dimension: int
    max_length: int


MODELS = [
    ModelConfig(
        name="BGE-large-zh-v1.5",
        model_id="BAAI/bge-large-zh-v1.5",
        dimension=1024,
        max_length=512,
    ),
    ModelConfig(
        name="BGE-M3",
        model_id="BAAI/bge-m3",
        dimension=1024,
        max_length=8192,
    ),
    ModelConfig(
        name="multilingual-e5-large",
        model_id="intfloat/multilingual-e5-large-instruct",
        dimension=1024,
        max_length=512,
    ),
]


# ============================================================
# 基准测试
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def benchmark_model(config: ModelConfig):
    """
    对单个模型运行完整基准测试。
    """
    print(f"\n{'=' * 60}")
    print(f"Testing: {config.name} ({config.model_id})")
    print(f"Dimension: {config.dimension}, Max Length: {config.max_length}")
    print(f"{'=' * 60}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [跳过] 需要安装 sentence-transformers: pip install sentence-transformers")
        return None

    # 加载模型
    print(f"  Loading model...")
    load_start = time.time()
    try:
        model = SentenceTransformer(config.model_id)
    except Exception as e:
        print(f"  [跳过] 加载失败: {e}")
        return None
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s")

    results = {"name": config.name, "dimension": config.dimension}

    # --- 1. 单条编码速度 ---
    print(f"\n  [1/3] 单条编码速度...")
    single_text = "向量数据库是用于高效存储和检索高维向量数据的数据库系统。"

    times = []
    for _ in range(10):
        start = time.time()
        _ = model.encode(single_text, normalize_embeddings=True)
        times.append(time.time() - start)

    avg_single = np.mean(times) * 1000
    results["single_encode_ms"] = round(avg_single, 1)
    print(f"  单条编码: {avg_single:.1f}ms (avg of 10)")

    # --- 2. 批量编码速度 ---
    print(f"  [2/3] 批量编码速度 (100 条)...")
    start = time.time()
    batch_embeddings = model.encode(BATCH_TEXTS, normalize_embeddings=True, batch_size=32)
    batch_time = time.time() - start

    results["batch_100_seconds"] = round(batch_time, 2)
    results["batch_qps"] = round(100 / batch_time, 1)
    print(f"  100 条编码: {batch_time:.2f}s ({results['batch_qps']} QPS)")

    # --- 3. 语义相似度质量 ---
    print(f"  [3/3] 语义相似度质量...")
    correct = 0
    total = len(SIMILARITY_PAIRS)

    for sent_a, sent_b, expected in SIMILARITY_PAIRS:
        emb_a = model.encode(sent_a, normalize_embeddings=True)
        emb_b = model.encode(sent_b, normalize_embeddings=True)
        sim = cosine_similarity(emb_a, emb_b)

        # 简单判断：相似度 > 0.7 为 high，< 0.5 为 low
        if expected == "high" and sim > 0.7:
            correct += 1
        elif expected == "low" and sim < 0.5:
            correct += 1

    accuracy = correct / total
    results["similarity_accuracy"] = f"{accuracy:.0%}"
    print(f"  相似度判断准确率: {accuracy:.0%} ({correct}/{total})")

    return results


def print_comparison(results: list[dict]):
    """打印对比表格"""
    if not results:
        print("\n没有可用的测试结果。")
        return

    print(f"\n\n{'=' * 70}")
    print("Embedding 模型对比结果")
    print(f"{'=' * 70}")

    # 表头
    headers = ["模型", "维度", "单条(ms)", "批量QPS", "准确率"]
    widths = [25, 6, 10, 10, 8]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    for r in results:
        row = [
            r["name"][:25].ljust(25),
            str(r["dimension"]).ljust(6),
            str(r.get("single_encode_ms", "N/A")).ljust(10),
            str(r.get("batch_qps", "N/A")).ljust(10),
            str(r.get("similarity_accuracy", "N/A")).ljust(8),
        ]
        print(" | ".join(row))

    print(f"\n选型建议：")
    print(f"  - 纯中文 + 追求速度 → BGE-large-zh-v1.5")
    print(f"  - 中英混合 + Hybrid Search → BGE-M3")
    print(f"  - 多语言通用 → multilingual-e5-large")


# ============================================================
# 维度截断实验（Matryoshka Embedding）
# ============================================================

def dimension_tradeoff_experiment():
    """
    测试不同向量维度对检索效果的影响。
    使用 Matryoshka 风格的截断：直接取前 N 维。
    """
    print(f"\n\n{'=' * 60}")
    print("向量维度 Tradeoff 实验")
    print(f"{'=' * 60}")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    except Exception as e:
        print(f"  [跳过] {e}")
        return

    # 编码所有测试对
    all_texts = []
    for a, b, _ in SIMILARITY_PAIRS:
        all_texts.extend([a, b])

    embeddings_full = model.encode(all_texts, normalize_embeddings=True)
    full_dim = embeddings_full.shape[1]

    dimensions_to_test = [64, 128, 256, 512, full_dim]

    print(f"\n原始维度: {full_dim}")
    print(f"{'维度':<10} {'相似度判断准确率':<20} {'存储大小比例':<15}")
    print("-" * 45)

    for dim in dimensions_to_test:
        if dim > full_dim:
            continue

        # 截断到指定维度
        truncated = embeddings_full[:, :dim]
        # 重新归一化
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        truncated = truncated / norms

        correct = 0
        for i, (_, _, expected) in enumerate(SIMILARITY_PAIRS):
            emb_a = truncated[i * 2]
            emb_b = truncated[i * 2 + 1]
            sim = cosine_similarity(emb_a, emb_b)

            if expected == "high" and sim > 0.7:
                correct += 1
            elif expected == "low" and sim < 0.5:
                correct += 1

        accuracy = correct / len(SIMILARITY_PAIRS)
        storage_ratio = dim / full_dim

        print(f"{dim:<10} {accuracy:<20.0%} {storage_ratio:<15.1%}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("Embedding 模型基准测试")
    print("注意：需要下载模型权重，首次运行可能较慢\n")

    # 逐个测试模型
    results = []
    for config in MODELS:
        result = benchmark_model(config)
        if result:
            results.append(result)

    # 打印对比
    print_comparison(results)

    # 维度实验
    dimension_tradeoff_experiment()


if __name__ == "__main__":
    main()
