"""
Qdrant 向量数据库快速上手
演示：创建 collection、插入、检索、更新、删除、过滤查询。

前提：
  docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.12.5

运行：
  pip install -r requirements.txt
  python 02_vector_db_quickstart.py
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_db")


# ============================================================
# Qdrant 基础操作封装
# ============================================================

class VectorStore:
    """
    Qdrant 向量存储的轻量封装。
    生产环境建议直接用 qdrant_client 的原生 API。
    """

    def __init__(self, host: str = "localhost", port: int = 6333):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=host, port=port)
        logger.info(f"Connected to Qdrant at {host}:{port}")

    def create_collection(
        self,
        name: str,
        dimension: int = 1024,
        distance: str = "cosine",
        on_disk: bool = False,
    ):
        """
        创建 collection。

        Args:
            name: collection 名称
            dimension: 向量维度
            distance: 距离度量 (cosine / euclid / dot)
            on_disk: 是否将向量存储在磁盘（大数据量时节省内存）
        """
        from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }

        self.client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dimension,
                distance=distance_map[distance],
                on_disk=on_disk,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,                # HNSW 图的连接数（默认 16，越大越精确但越慢）
                ef_construct=128,    # 构建时搜索宽度（默认 100）
            ),
        )
        logger.info(f"Collection '{name}' created (dim={dimension}, distance={distance})")

    def upsert(
        self,
        collection: str,
        vectors: list[list[float]],
        payloads: list[dict],
        ids: Optional[list[str]] = None,
    ) -> int:
        """
        插入或更新向量。

        Args:
            collection: collection 名称
            vectors: 向量列表
            payloads: 元数据列表
            ids: 可选的 ID 列表

        Returns:
            插入的数量
        """
        from qdrant_client.models import PointStruct

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        points = [
            PointStruct(id=id_, vector=vec, payload=payload)
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]

        # 分批插入（每批 100 条，避免单次请求太大）
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=collection, points=batch)

        logger.info(f"Upserted {len(points)} points to '{collection}'")
        return len(points)

    def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[dict] = None,
    ) -> list[dict]:
        """
        向量检索。

        Args:
            collection: collection 名称
            query_vector: 查询向量
            limit: 返回数量
            score_threshold: 最低分数阈值
            filter_conditions: 过滤条件

        Returns:
            检索结果列表 [{"id", "score", "payload"}, ...]
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # 构建过滤条件
        qdrant_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            qdrant_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
            }
            for r in results.points
        ]

    def delete(self, collection: str, ids: list[str]):
        """删除指定 ID 的向量"""
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=ids),
        )
        logger.info(f"Deleted {len(ids)} points from '{collection}'")

    def get_collection_info(self, collection: str) -> dict:
        """获取 collection 信息"""
        info = self.client.get_collection(collection)
        return {
            "name": collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
            "config": {
                "dimension": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value,
            },
        }


# ============================================================
# 演示
# ============================================================

def generate_dummy_embedding(dim: int = 1024) -> list[float]:
    """生成随机归一化向量（仅演示用，实际应用用 Embedding 模型）"""
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def demo():
    print("Qdrant 向量数据库快速上手\n")

    try:
        store = VectorStore(host="localhost", port=6333)
    except Exception as e:
        print(f"无法连接到 Qdrant: {e}")
        print("请先启动 Qdrant: docker run -d -p 6333:6333 qdrant/qdrant:v1.12.5")
        print("\n以下为操作演示（dry run）：\n")
        dry_run_demo()
        return

    collection = "demo_documents"

    # 1. 创建 collection
    print("1. 创建 Collection")
    store.create_collection(collection, dimension=1024, distance="cosine")

    # 2. 插入数据
    print("\n2. 插入 10 条文档")
    documents = [
        {"text": "Kubernetes 是一个容器编排平台", "source": "k8s-doc", "category": "infra"},
        {"text": "vLLM 是高性能的 LLM 推理引擎", "source": "vllm-doc", "category": "llm"},
        {"text": "向量数据库用于存储和检索嵌入向量", "source": "rag-doc", "category": "rag"},
        {"text": "Transformer 架构是现代 NLP 的基础", "source": "ml-doc", "category": "ml"},
        {"text": "Docker 容器化简化了应用部署", "source": "docker-doc", "category": "infra"},
        {"text": "Prometheus 用于监控和告警", "source": "monitor-doc", "category": "infra"},
        {"text": "RAG 系统结合了检索和生成", "source": "rag-doc", "category": "rag"},
        {"text": "GPU 调度是 LLM 部署的关键挑战", "source": "gpu-doc", "category": "llm"},
        {"text": "BM25 是经典的稀疏检索算法", "source": "search-doc", "category": "rag"},
        {"text": "CUDA 是 NVIDIA 的并行计算平台", "source": "gpu-doc", "category": "infra"},
    ]

    vectors = [generate_dummy_embedding(1024) for _ in documents]
    ids = [str(i) for i in range(len(documents))]
    store.upsert(collection, vectors, documents, ids)

    # 3. 基础检索
    print("\n3. 基础检索 (Top-3)")
    query_vec = generate_dummy_embedding(1024)
    results = store.search(collection, query_vec, limit=3)
    for r in results:
        print(f"  score={r['score']:.4f} | {r['payload']['text']}")

    # 4. 带过滤的检索
    print("\n4. 过滤检索 (category=rag)")
    results = store.search(
        collection, query_vec, limit=3,
        filter_conditions={"category": "rag"},
    )
    for r in results:
        print(f"  score={r['score']:.4f} | {r['payload']['text']}")

    # 5. 查看 collection 信息
    print("\n5. Collection 信息")
    info = store.get_collection_info(collection)
    for k, v in info.items():
        print(f"  {k}: {v}")

    # 6. 删除
    print("\n6. 删除文档 (id=0)")
    store.delete(collection, ["0"])
    info = store.get_collection_info(collection)
    print(f"  删除后 points_count: {info['points_count']}")

    # 7. 性能测试
    print("\n7. 批量插入性能测试 (1000 条)")
    large_vectors = [generate_dummy_embedding(1024) for _ in range(1000)]
    large_payloads = [{"text": f"doc-{i}", "idx": i} for i in range(1000)]

    start = time.time()
    store.upsert(collection, large_vectors, large_payloads)
    elapsed = time.time() - start
    print(f"  1000 条插入耗时: {elapsed:.2f}s ({1000/elapsed:.0f} docs/s)")

    # 检索性能
    times = []
    for _ in range(100):
        q = generate_dummy_embedding(1024)
        start = time.time()
        _ = store.search(collection, q, limit=10)
        times.append(time.time() - start)

    avg_ms = np.mean(times) * 1000
    p99_ms = np.percentile(times, 99) * 1000
    print(f"  检索延迟 (100 次): avg={avg_ms:.1f}ms, P99={p99_ms:.1f}ms")

    print("\n完成！")


def dry_run_demo():
    """Qdrant 不可用时的 dry run 演示"""
    print("--- Qdrant 操作流程（伪代码）---\n")
    print("""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(host="localhost", port=6333)

# 创建 collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# 插入
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(id=1, vector=[0.1, 0.2, ...], payload={"text": "..."}),
    ],
)

# 检索
results = client.query_points(
    collection_name="documents",
    query=[0.1, 0.2, ...],
    limit=10,
)

# 带过滤的检索
from qdrant_client.models import Filter, FieldCondition, MatchValue
results = client.query_points(
    collection_name="documents",
    query=[0.1, 0.2, ...],
    query_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="rag"))]
    ),
    limit=10,
)
""")


if __name__ == "__main__":
    demo()
