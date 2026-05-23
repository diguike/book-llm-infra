"""
完整 RAG Pipeline 示例
从文档加载 → Chunking → Embedding → 存储 → 检索 → Reranking → 生成

运行：
  pip install -r requirements.txt
  python 05_rag_pipeline.py

默认使用模拟的 Embedding 和 LLM（无需 GPU），
设置环境变量启用真实模型：
  USE_REAL_MODELS=1 python 05_rag_pipeline.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_pipeline")

USE_REAL_MODELS = os.getenv("USE_REAL_MODELS", "0") == "1"


# ============================================================
# 数据模型
# ============================================================

@dataclass
class Document:
    """原始文档"""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """文档片段"""
    id: str
    text: str
    doc_id: str
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """检索结果"""
    chunk: Chunk
    score: float
    rerank_score: float | None = None


@dataclass
class RAGResponse:
    """RAG 响应"""
    answer: str
    sources: list[SearchResult]
    metrics: dict = field(default_factory=dict)


# ============================================================
# 1. Chunking
# ============================================================

class ChunkingService:
    """文档切分服务"""

    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """对单个文档进行切分"""
        text = doc.text.strip()
        if not text:
            return []

        # 按段落 → 句子的优先级切分
        raw_chunks = self._recursive_split(text)

        chunks = []
        for i, text_chunk in enumerate(raw_chunks):
            chunk_id = f"{doc.id}_chunk_{i}"
            chunks.append(Chunk(
                id=chunk_id,
                text=text_chunk,
                doc_id=doc.id,
                metadata={**doc.metadata, "chunk_index": i},
            ))
        return chunks

    def _recursive_split(self, text: str) -> list[str]:
        separators = ["\n\n", "\n", "。", "！", "？", "；", " "]

        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # 找最佳分隔符
        for sep in separators:
            if sep in text:
                return self._split_by_sep(text, sep)

        # 最后按字符切
        return self._fixed_split(text)

    def _split_by_sep(self, text: str, sep: str) -> list[str]:
        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(part) > self.chunk_size:
                    chunks.extend(self._fixed_split(part))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _fixed_split(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks


# ============================================================
# 2. Embedding
# ============================================================

class EmbeddingService:
    """Embedding 服务，支持缓存"""

    def __init__(self):
        self._model = None
        self._cache: dict[str, list[float]] = {}
        self.dimension = 128  # 默认用模拟维度

    def _get_model(self):
        if self._model is not None:
            return self._model
        if USE_REAL_MODELS:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
                self.dimension = 1024
                logger.info("Loaded real embedding model")
                return self._model
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
        return None

    def embed(self, text: str) -> list[float]:
        """单条文本 embedding（带缓存）"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        model = self._get_model()
        if model:
            vec = model.encode(text, normalize_embeddings=True).tolist()
        else:
            # 模拟 embedding
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dimension).astype(np.float32)
            vec = (vec / np.linalg.norm(vec)).tolist()

        self._cache[cache_key] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding"""
        # 分离缓存命中和未命中的
        results = [None] * len(texts)
        to_encode = []
        to_encode_indices = []

        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                to_encode.append(text)
                to_encode_indices.append(i)

        if to_encode:
            model = self._get_model()
            if model:
                vecs = model.encode(to_encode, normalize_embeddings=True, batch_size=32)
                for idx, vec in zip(to_encode_indices, vecs):
                    vec_list = vec.tolist()
                    results[idx] = vec_list
                    cache_key = hashlib.md5(to_encode[to_encode_indices.index(idx)].encode()).hexdigest()
                    self._cache[cache_key] = vec_list
            else:
                for i, text in zip(to_encode_indices, to_encode):
                    results[i] = self.embed(text)

        return results

    @property
    def cache_stats(self) -> dict:
        return {"cache_size": len(self._cache)}


# ============================================================
# 3. Vector Store (in-memory 简化版)
# ============================================================

class InMemoryVectorStore:
    """
    内存向量存储（演示用）。
    生产环境替换为 Qdrant / Milvus / pgvector。
    """

    def __init__(self):
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, np.ndarray] = {}

    def add(self, chunks: list[Chunk]):
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")
            self.chunks[chunk.id] = chunk
            self.vectors[chunk.id] = np.array(chunk.embedding)

    def search(self, query_vector: list[float], top_k: int = 10) -> list[tuple[Chunk, float]]:
        if not self.vectors:
            return []

        q = np.array(query_vector)
        scores = {}
        for chunk_id, vec in self.vectors.items():
            scores[chunk_id] = float(np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec)))

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [(self.chunks[cid], scores[cid]) for cid in sorted_ids]

    @property
    def size(self) -> int:
        return len(self.chunks)


# ============================================================
# 4. Reranker
# ============================================================

class Reranker:
    """
    Cross-Encoder Reranker。
    用 Cross-Encoder 对检索结果重排序，提升精度。
    """

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        if USE_REAL_MODELS:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024)
                logger.info("Loaded reranker model")
                return self._model
            except Exception:
                pass
        return None

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 5,
    ) -> list[SearchResult]:
        """对检索结果重排序"""
        if not results:
            return []

        model = self._get_model()
        if model:
            pairs = [(query, r.chunk.text) for r in results]
            scores = model.predict(pairs)
            for r, score in zip(results, scores):
                r.rerank_score = float(score)
        else:
            # 模拟：用原始分数 + 少量随机扰动
            for r in results:
                r.rerank_score = r.score + np.random.uniform(-0.05, 0.05)

        # 按 rerank 分数排序
        results.sort(key=lambda r: r.rerank_score or 0, reverse=True)
        return results[:top_n]


# ============================================================
# 5. LLM 生成
# ============================================================

class LLMService:
    """LLM 服务"""

    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """调用 LLM 生成回答"""
        if USE_REAL_MODELS:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{self.backend_url}/v1/chat/completions",
                        json={
                            "model": "default",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                        },
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"LLM call failed: {e}, using mock response")

        # 模拟生成
        return f"[模拟回答] 根据提供的上下文，这是一个关于该主题的综合回答。（实际部署时连接 vLLM 后端）"


# ============================================================
# 6. RAG Pipeline
# ============================================================

class RAGPipeline:
    """
    完整的 RAG Pipeline。

    流程：
      Query → Embedding → Vector Search → Reranking → Prompt 构造 → LLM 生成
    """

    def __init__(
        self,
        embedding: EmbeddingService,
        vector_store: InMemoryVectorStore,
        reranker: Reranker,
        llm: LLMService,
        retrieve_top_k: int = 20,
        rerank_top_n: int = 5,
    ):
        self.embedding = embedding
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_n = rerank_top_n

    def ingest(self, documents: list[Document]):
        """摄入文档：切分 → Embedding → 存储"""
        chunker = ChunkingService(chunk_size=400, overlap=50)

        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")

        # 批量 Embedding
        texts = [c.text for c in all_chunks]
        embeddings = self.embedding.embed_batch(texts)

        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        # 存入向量库
        self.vector_store.add(all_chunks)
        logger.info(f"Indexed {len(all_chunks)} chunks into vector store")

    async def query(self, question: str) -> RAGResponse:
        """执行 RAG 查询"""
        metrics = {}
        total_start = time.time()

        # Step 1: Embedding 查询
        t0 = time.time()
        query_vector = self.embedding.embed(question)
        metrics["embedding_ms"] = round((time.time() - t0) * 1000, 1)

        # Step 2: 向量检索
        t0 = time.time()
        raw_results = self.vector_store.search(query_vector, top_k=self.retrieve_top_k)
        search_results = [
            SearchResult(chunk=chunk, score=score)
            for chunk, score in raw_results
        ]
        metrics["retrieval_ms"] = round((time.time() - t0) * 1000, 1)
        metrics["candidates"] = len(search_results)

        # Step 3: Reranking
        t0 = time.time()
        reranked = self.reranker.rerank(question, search_results, top_n=self.rerank_top_n)
        metrics["reranking_ms"] = round((time.time() - t0) * 1000, 1)

        # Step 4: 构造 Prompt
        context = "\n\n---\n\n".join([
            f"[来源: {r.chunk.metadata.get('source', 'unknown')}]\n{r.chunk.text}"
            for r in reranked
        ])

        prompt = f"""根据以下参考资料回答问题。如果资料中没有相关信息，请如实说明。

参考资料：
{context}

问题：{question}

回答："""

        # Step 5: LLM 生成
        t0 = time.time()
        answer = await self.llm.generate(prompt)
        metrics["generation_ms"] = round((time.time() - t0) * 1000, 1)

        metrics["total_ms"] = round((time.time() - total_start) * 1000, 1)
        metrics["prompt_length"] = len(prompt)

        return RAGResponse(
            answer=answer,
            sources=reranked,
            metrics=metrics,
        )


# ============================================================
# 演示
# ============================================================

# 示例文档
SAMPLE_DOCUMENTS = [
    Document(
        id="doc-vllm",
        text="""vLLM 是由加州大学伯克利分校开发的高性能大语言模型推理引擎。它通过 PagedAttention 技术革新了 KV Cache 的管理方式，将显存浪费从传统方案的 60-80% 降低到 4% 以下。

vLLM 的核心特性包括：
1. Continuous Batching：动态合并不同阶段的请求，最大化 GPU 利用率
2. PagedAttention：借鉴操作系统虚拟内存的思想，分页管理 KV Cache
3. OpenAI 兼容 API：可以直接替代 OpenAI 的 API 端点
4. 多模型支持：LLaMA、Qwen、Mistral、ChatGLM 等主流模型

在生产部署方面，vLLM 提供了官方 Docker 镜像，支持通过环境变量配置模型参数。配合 Kubernetes 使用时，需要注意 GPU 资源声明、共享内存配置和健康检查超时设置。""",
        metadata={"source": "llm-infra-book/ch10", "topic": "inference"},
    ),
    Document(
        id="doc-k8s-gpu",
        text="""在 Kubernetes 中调度 GPU 资源需要安装 NVIDIA device plugin。安装后，每个 GPU 节点会自动上报 nvidia.com/gpu 资源。

GPU 资源的声明方式：
- 在 Pod spec 的 resources.limits 中声明 nvidia.com/gpu: 1
- GPU 只能整数分配，不像 CPU 可以请求 0.5 核
- 建议为 GPU 节点设置 taint，防止非 GPU 工作负载占用昂贵资源

GPU 共享方案：
1. MIG (Multi-Instance GPU)：A100/H100 支持，将物理 GPU 切分为独立实例
2. Time-slicing：通过 device plugin 配置实现时间片共享，不需要特殊硬件

阿里云 ACK 的 GPU 节点池配置要点：
- 选择合适的实例类型（如 ecs.gn7i 系列对应 A10 GPU）
- 数据盘至少 500GB，用于存储模型权重
- 设置节点标签（如 gpu-type=a10）便于调度""",
        metadata={"source": "llm-infra-book/ch12", "topic": "deployment"},
    ),
    Document(
        id="doc-rag",
        text="""RAG（Retrieval-Augmented Generation）通过检索外部知识库来增强大语言模型的回答质量。一个生产级的 RAG 系统需要解决以下基础设施问题：

1. Embedding 模型选择与部署
推荐使用 BGE-M3 作为 Embedding 模型，它同时支持 Dense 和 Sparse 向量，一个模型就能支持 Hybrid Search。部署可以使用 HuggingFace TEI，相比手动加载 sentence-transformers 快 3-4 倍。

2. 向量数据库选型
- 小规模（< 100 万条）+ 已有 PostgreSQL → pgvector
- 中等规模（< 1000 万条）→ Qdrant
- 大规模（> 1000 万条）→ Milvus

3. Chunking 策略
推荐 300-500 字的 chunk 大小，使用 Recursive Character Splitter 保持语义完整性。对 Markdown 文档，按标题结构切分效果更好。

4. Hybrid Search
纯向量检索在精确关键词匹配场景下表现不佳。建议使用 Dense + Sparse 的混合检索，通过 RRF 融合两个结果列表。""",
        metadata={"source": "llm-infra-book/ch14", "topic": "rag"},
    ),
]


async def main():
    print("=" * 60)
    print("RAG Pipeline 完整示例")
    print("=" * 60)

    # 初始化组件
    embedding = EmbeddingService()
    vector_store = InMemoryVectorStore()
    reranker = Reranker()
    llm = LLMService()

    pipeline = RAGPipeline(
        embedding=embedding,
        vector_store=vector_store,
        reranker=reranker,
        llm=llm,
        retrieve_top_k=10,
        rerank_top_n=3,
    )

    # 摄入文档
    print("\n[1] 摄入文档...")
    pipeline.ingest(SAMPLE_DOCUMENTS)
    print(f"    向量库大小: {vector_store.size} chunks")
    print(f"    Embedding 缓存: {embedding.cache_stats}")

    # 查询
    questions = [
        "vLLM 有什么核心特性？",
        "如何在 K8s 中声明 GPU 资源？",
        "RAG 系统应该选什么向量数据库？",
        "Chunk 大小推荐多少？",
    ]

    print(f"\n[2] 查询测试 ({len(questions)} 个问题)...")

    for q in questions:
        print(f"\n{'─' * 50}")
        print(f"Q: {q}")

        response = await pipeline.query(q)

        print(f"A: {response.answer[:200]}...")
        print(f"\n  来源 ({len(response.sources)} 个):")
        for i, src in enumerate(response.sources):
            text_preview = src.chunk.text[:60].replace("\n", " ")
            print(f"    [{i+1}] score={src.score:.3f} "
                  f"rerank={src.rerank_score:.3f} "
                  f"| {text_preview}...")

        print(f"\n  性能指标:")
        for k, v in response.metrics.items():
            print(f"    {k}: {v}")

    # Pipeline 性能总结
    print(f"\n\n{'=' * 60}")
    print("Pipeline 架构总结")
    print(f"{'=' * 60}")
    print("""
  Query
    │
    ▼
  Embedding (20-50ms)
    │
    ▼
  Vector Search (10-30ms)          ← 可替换为 Qdrant / Milvus
    │ top-20 candidates
    ▼
  Reranking (50-200ms)             ← Cross-Encoder: bge-reranker-v2-m3
    │ top-5 results
    ▼
  Prompt Construction
    │
    ▼
  LLM Generation (200-2000ms+)    ← vLLM / 云 API
    │
    ▼
  Response

优化建议：
  1. Embedding 缓存：相同查询不重复计算
  2. Prefix Caching：System prompt + 检索上下文的 KV Cache 复用
  3. Streaming：先返回第一个 token，再逐步输出
  4. 异步检索：多个 collection 并行检索
""")


if __name__ == "__main__":
    asyncio.run(main())
