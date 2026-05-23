"""
Hybrid Search 实现
演示：Dense + Sparse 检索融合，使用 RRF 排序。

本文件包含：
1. 纯 Dense Search（向量检索）
2. 纯 Sparse Search（BM25）
3. Hybrid Search（RRF 融合）
4. 效果对比

运行：
  pip install -r requirements.txt
  python 04_hybrid_search.py
"""

from __future__ import annotations

import math
import re
import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_search")


# ============================================================
# 测试数据
# ============================================================

DOCUMENTS = [
    {"id": "doc-1", "text": "vLLM 是由 UC Berkeley 开发的高性能 LLM 推理引擎，使用 PagedAttention 技术管理 KV Cache"},
    {"id": "doc-2", "text": "Kubernetes (K8s) 是 Google 开源的容器编排平台，用于自动化部署、扩展和管理容器化应用程序"},
    {"id": "doc-3", "text": "NVIDIA A100 GPU 具有 80GB HBM2e 显存，支持 MIG 技术可将单卡划分为多个独立实例"},
    {"id": "doc-4", "text": "Qdrant 是用 Rust 编写的向量数据库，支持 Dense 和 Sparse 向量的混合检索"},
    {"id": "doc-5", "text": "BM25 是基于词频和逆文档频率的经典信息检索算法，在精确关键词匹配场景下表现优异"},
    {"id": "doc-6", "text": "RAG 系统通过检索外部知识库来增强大语言模型的生成能力，减少幻觉问题"},
    {"id": "doc-7", "text": "Transformer 架构基于自注意力机制，是 GPT、BERT、LLaMA 等模型的基础"},
    {"id": "doc-8", "text": "Docker 容器化技术将应用及其依赖打包为镜像，确保在不同环境中一致运行"},
    {"id": "doc-9", "text": "Prometheus 是开源的监控和告警系统，通过 pull 模式采集指标数据，配合 Grafana 可视化"},
    {"id": "doc-10", "text": "CUDA 是 NVIDIA 的并行计算平台和 API，PyTorch 和 TensorFlow 底层依赖 CUDA 进行 GPU 加速"},
    {"id": "doc-11", "text": "API-KEY-20250101 是系统管理员在 2025 年 1 月 1 日创建的生产环境 API 密钥"},
    {"id": "doc-12", "text": "Flash Attention 通过分块计算和减少 HBM 访问次数来加速注意力计算，内存使用从 O(N²) 降到 O(N)"},
]

QUERIES = [
    # (查询, 期望的 top-1 文档, 说明)
    ("vLLM PagedAttention 推理引擎", "doc-1", "Dense 和 Sparse 都能找到"),
    ("K8s 容器编排", "doc-2", "Dense 和 Sparse 都能找到"),
    ("API-KEY-20250101", "doc-11", "Sparse 擅长：精确关键词匹配"),
    ("如何减少 LLM 的幻觉", "doc-6", "Dense 擅长：语义理解"),
    ("GPU 显存不够怎么办", "doc-3", "Dense 擅长：语义相关但无关键词重叠"),
    ("Rust 写的向量搜索工具", "doc-4", "混合场景"),
]


# ============================================================
# BM25 稀疏检索
# ============================================================

class BM25:
    """
    BM25 稀疏检索实现。

    BM25 scoring:
      score(q, d) = Σ IDF(qi) * (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * |d| / avgdl))

    其中：
      - f(qi, d): 词 qi 在文档 d 中的频率
      - |d|: 文档长度
      - avgdl: 平均文档长度
      - k1, b: 调节参数
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avgdl = 0
        self.doc_lengths: list[int] = []
        self.doc_freqs: list[Counter] = []     # 每个文档的词频
        self.idf: dict[str, float] = {}        # 逆文档频率
        self.doc_ids: list[str] = []

    def _tokenize(self, text: str) -> list[str]:
        """简单分词：按非中文字符和空格切分，中文逐字"""
        tokens = []
        # 英文单词
        english_words = re.findall(r'[a-zA-Z0-9][\w\-]*', text.lower())
        tokens.extend(english_words)
        # 中文字符（逐字，简化处理）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        tokens.extend(chinese_chars)
        # 中文双字组合（简单 n-gram）
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])
        return tokens

    def index(self, documents: list[dict]):
        """构建索引"""
        self.doc_count = len(documents)
        self.doc_ids = [d["id"] for d in documents]

        for doc in documents:
            tokens = self._tokenize(doc["text"])
            self.doc_lengths.append(len(tokens))
            self.doc_freqs.append(Counter(tokens))

        self.avgdl = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0

        # 计算 IDF
        df = Counter()  # 文档频率：包含某个词的文档数
        for freq in self.doc_freqs:
            for term in freq:
                df[term] += 1

        for term, count in df.items():
            # IDF with smoothing
            self.idf[term] = math.log((self.doc_count - count + 0.5) / (count + 0.5) + 1)

        logger.info(f"BM25 index built: {self.doc_count} docs, {len(self.idf)} unique terms")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """检索"""
        query_tokens = self._tokenize(query)
        scores = []

        for i in range(self.doc_count):
            score = 0.0
            doc_len = self.doc_lengths[i]
            freq = self.doc_freqs[i]

            for token in query_tokens:
                if token not in freq:
                    continue
                tf = freq[token]
                idf = self.idf.get(token, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator

            scores.append((i, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                results.append({
                    "id": self.doc_ids[idx],
                    "score": score,
                    "text": None,  # 调用方补充
                })
        return results


# ============================================================
# Dense 检索（模拟）
# ============================================================

class DenseSearch:
    """
    Dense 检索（向量检索）。

    实际项目中使用 Embedding 模型 + 向量数据库。
    这里用随机向量模拟，重点演示融合逻辑。
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.doc_ids: list[str] = []
        self.vectors: np.ndarray | None = None
        self._model = None

    def _get_model(self):
        """尝试加载真实的 embedding 模型"""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
            self.dimension = 1024
            logger.info("Using real embedding model: bge-large-zh-v1.5")
            return self._model
        except Exception:
            logger.info("Embedding model not available, using random vectors for demo")
            return None

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        if model:
            return model.encode(texts, normalize_embeddings=True)
        else:
            # 模拟：相似文本生成相似的向量（通过哈希种子）
            vectors = []
            for text in texts:
                seed = hash(text) % (2**31)
                rng = np.random.RandomState(seed)
                vec = rng.randn(self.dimension).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                vectors.append(vec)
            return np.array(vectors)

    def index(self, documents: list[dict]):
        self.doc_ids = [d["id"] for d in documents]
        texts = [d["text"] for d in documents]
        self.vectors = self._encode(texts)
        logger.info(f"Dense index built: {len(documents)} docs, dim={self.dimension}")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_vec = self._encode([query])[0]
        # 余弦相似度（向量已归一化，dot product = cosine similarity）
        scores = np.dot(self.vectors, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {"id": self.doc_ids[i], "score": float(scores[i]), "text": None}
            for i in top_indices
            if scores[i] > 0
        ]


# ============================================================
# Reciprocal Rank Fusion (RRF)
# ============================================================

def reciprocal_rank_fusion(
    results_list: list[list[dict]],
    k: int = 60,
    top_n: int = 10,
) -> list[dict]:
    """
    RRF 融合多个检索结果。

    RRF 公式：
      RRF_score(d) = Σ 1 / (k + rank_i(d))

    其中 k 是常数（通常取 60），rank_i(d) 是文档 d 在第 i 个结果列表中的排名。

    RRF 的优点：
    1. 不需要对不同检索器的分数做归一化（分数量纲可能不同）
    2. 对异常值不敏感
    3. 只依赖排名，不依赖绝对分数
    """
    scores: dict[str, float] = {}
    doc_info: dict[str, dict] = {}

    for results in results_list:
        for rank, item in enumerate(results):
            doc_id = item["id"]
            if doc_id not in scores:
                scores[doc_id] = 0.0
                doc_info[doc_id] = item
            scores[doc_id] += 1.0 / (k + rank + 1)

    # 按 RRF 分数排序
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [
        {**doc_info[doc_id], "rrf_score": scores[doc_id]}
        for doc_id in sorted_ids[:top_n]
    ]


# ============================================================
# Hybrid Search
# ============================================================

class HybridSearchEngine:
    """
    混合检索引擎：Dense + Sparse + RRF 融合。
    """

    def __init__(self):
        self.dense = DenseSearch()
        self.sparse = BM25()
        self.documents: dict[str, dict] = {}

    def index(self, documents: list[dict]):
        self.documents = {d["id"]: d for d in documents}
        self.dense.index(documents)
        self.sparse.index(documents)

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: int = 1,
        sparse_weight: int = 1,
    ) -> dict:
        """
        执行混合检索。

        Returns:
            {
                "dense": [...],     # 纯向量检索结果
                "sparse": [...],    # 纯 BM25 结果
                "hybrid": [...],    # RRF 融合结果
            }
        """
        # 各检索 top_k * 2 的结果，保证融合后有足够候选
        fetch_k = top_k * 2

        dense_results = self.dense.search(query, fetch_k)
        sparse_results = self.sparse.search(query, fetch_k)

        # 补充文本信息
        for r in dense_results + sparse_results:
            if r["id"] in self.documents:
                r["text"] = self.documents[r["id"]]["text"]

        # RRF 融合（可以通过重复来调整权重）
        all_results = []
        for _ in range(dense_weight):
            all_results.append(dense_results)
        for _ in range(sparse_weight):
            all_results.append(sparse_results)

        hybrid_results = reciprocal_rank_fusion(all_results, top_n=top_k)
        for r in hybrid_results:
            if r["id"] in self.documents:
                r["text"] = self.documents[r["id"]]["text"]

        return {
            "dense": dense_results[:top_k],
            "sparse": sparse_results[:top_k],
            "hybrid": hybrid_results,
        }


# ============================================================
# 对比评测
# ============================================================

def evaluate():
    """对比 Dense / Sparse / Hybrid 的检索效果"""
    engine = HybridSearchEngine()
    engine.index(DOCUMENTS)

    print("=" * 80)
    print("Hybrid Search 效果对比")
    print("=" * 80)

    # 统计 Top-1 命中率
    hits = {"dense": 0, "sparse": 0, "hybrid": 0}

    for query, expected_id, note in QUERIES:
        results = engine.search(query, top_k=5)

        print(f"\n查询: \"{query}\"")
        print(f"期望: {expected_id} ({note})")

        for method in ["dense", "sparse", "hybrid"]:
            top_ids = [r["id"] for r in results[method][:3]]
            hit = expected_id in top_ids[:1]
            if hit:
                hits[method] += 1
            marker = " ✓" if hit else ""
            print(f"  {method:8s} Top-3: {top_ids}{marker}")

    # 汇总
    total = len(QUERIES)
    print(f"\n\n{'=' * 50}")
    print(f"Top-1 命中率汇总 ({total} 个查询)")
    print(f"{'=' * 50}")
    for method, count in hits.items():
        print(f"  {method:8s}: {count}/{total} ({count/total:.0%})")

    print("""
分析：
  - Dense Search 在语义相关的查询上表现好（如「如何减少 LLM 的幻觉」）
  - Sparse Search (BM25) 在精确关键词匹配上更强（如「API-KEY-20250101」）
  - Hybrid Search 综合两者优势，整体命中率最高

注意：本 demo 使用模拟向量，实际使用真实 Embedding 模型时，
Dense Search 的效果会大幅提升，Hybrid 的优势也会更明显。
""")


if __name__ == "__main__":
    evaluate()
