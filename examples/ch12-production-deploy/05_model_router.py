"""
多模型路由器
功能：内容分类路由、Fallback 降级、成本感知溢出

运行：
  pip install -r requirements.txt
  python 05_model_router.py

本文件可独立运行演示路由逻辑，也可集成到 04_api_gateway.py 中。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_router")


# ============================================================
# 配置
# ============================================================

class ModelTier(Enum):
    LARGE = "large"     # 72B，复杂任务
    SMALL = "small"     # 7B，简单任务
    CLOUD = "cloud"     # 云 API 兜底


@dataclass
class BackendConfig:
    """后端配置"""
    name: str
    url: str
    model: str
    tier: ModelTier
    max_pending: int = 50                # 队列深度上限
    cost_per_mtok: float = 0.0           # 每百万 token 成本（元）
    healthy: bool = True
    last_health_check: float = 0.0
    pending_count: int = 0


# 后端列表
BACKENDS = [
    BackendConfig(
        name="qwen-72b-local",
        url="http://vllm-72b:8000",
        model="Qwen/Qwen2.5-72B-Instruct",
        tier=ModelTier.LARGE,
        max_pending=30,
        cost_per_mtok=8.0,    # 自建成本
    ),
    BackendConfig(
        name="qwen-7b-local",
        url="http://vllm-7b:8000",
        model="Qwen/Qwen2.5-7B-Instruct",
        tier=ModelTier.SMALL,
        max_pending=50,
        cost_per_mtok=2.5,    # 自建成本
    ),
    BackendConfig(
        name="siliconflow-7b",
        url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen2.5-7B-Instruct",
        tier=ModelTier.CLOUD,
        max_pending=999,      # 云 API 不限队列
        cost_per_mtok=0.35,   # 按量付费
    ),
]

# 复杂任务的关键词（简单规则分类）
COMPLEX_KEYWORDS = [
    "代码", "code", "编程", "debug", "分析", "analyze",
    "推理", "reason", "证明", "prove", "比较", "对比",
    "翻译长文", "总结", "summarize",
]


# ============================================================
# 请求分类器
# ============================================================

def classify_request(messages: list[dict]) -> ModelTier:
    """
    根据请求内容判断应该用大模型还是小模型。

    分类逻辑：
    1. 长上下文（> 2000 tokens 估算）→ 大模型
    2. 包含复杂任务关键词 → 大模型
    3. 其他 → 小模型
    """
    if not messages:
        return ModelTier.SMALL

    last_content = messages[-1].get("content", "").lower()
    total_chars = sum(len(m.get("content", "")) for m in messages)
    estimated_tokens = total_chars // 2  # 中文粗估：1 token ≈ 2 字符

    # 长上下文
    if estimated_tokens > 2000:
        logger.info(f"Classified as LARGE: estimated {estimated_tokens} tokens")
        return ModelTier.LARGE

    # 关键词匹配
    for kw in COMPLEX_KEYWORDS:
        if kw in last_content:
            logger.info(f"Classified as LARGE: matched keyword '{kw}'")
            return ModelTier.LARGE

    return ModelTier.SMALL


# ============================================================
# 路由器
# ============================================================

class ModelRouter:
    """
    多模型路由器

    路由策略：
    1. 根据请求内容分类（大模型 / 小模型）
    2. 优先使用自建后端
    3. 自建后端满载时溢出到云 API
    4. 主后端不健康时降级到 fallback
    """

    def __init__(self, backends: list[BackendConfig]):
        self.backends = {b.name: b for b in backends}
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        return self._http_client

    async def close(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    # ----- 健康检查 -----

    async def health_check(self, backend: BackendConfig) -> bool:
        """检查后端健康状态，每 30 秒最多检查一次"""
        now = time.time()
        if now - backend.last_health_check < 30:
            return backend.healthy

        try:
            client = await self._get_client()
            resp = await client.get(f"{backend.url}/health", timeout=3.0)
            backend.healthy = resp.status_code == 200
        except Exception:
            backend.healthy = False

        backend.last_health_check = now
        return backend.healthy

    # ----- 路由选择 -----

    def _get_backends_by_tier(self, tier: ModelTier) -> list[BackendConfig]:
        """获取指定 tier 的后端列表"""
        return [b for b in self.backends.values() if b.tier == tier]

    def _get_fallback_chain(self, tier: ModelTier) -> list[ModelTier]:
        """
        降级链：
        LARGE → SMALL → CLOUD
        SMALL → CLOUD
        """
        chains = {
            ModelTier.LARGE: [ModelTier.LARGE, ModelTier.SMALL, ModelTier.CLOUD],
            ModelTier.SMALL: [ModelTier.SMALL, ModelTier.CLOUD],
            ModelTier.CLOUD: [ModelTier.CLOUD],
        }
        return chains[tier]

    async def route(self, messages: list[dict]) -> BackendConfig:
        """
        核心路由逻辑

        Returns:
            选中的后端配置
        Raises:
            RuntimeError: 所有后端都不可用
        """
        tier = classify_request(messages)
        fallback_chain = self._get_fallback_chain(tier)

        for fallback_tier in fallback_chain:
            backends = self._get_backends_by_tier(fallback_tier)
            for backend in backends:
                # 跳过不健康的后端
                if not await self.health_check(backend):
                    logger.warning(f"Backend {backend.name} is unhealthy, skipping")
                    continue

                # 跳过满载的后端
                if backend.pending_count >= backend.max_pending:
                    logger.warning(
                        f"Backend {backend.name} is full "
                        f"({backend.pending_count}/{backend.max_pending}), skipping"
                    )
                    continue

                logger.info(
                    f"Selected backend: {backend.name} "
                    f"(tier={fallback_tier.value}, pending={backend.pending_count})"
                )
                return backend

        raise RuntimeError("All backends are unavailable")

    # ----- 成本感知路由 -----

    async def cost_aware_route(self, messages: list[dict]) -> BackendConfig:
        """
        成本感知路由：优先自建（成本低），满载溢出到云 API（弹性好）

        与普通 route() 的区别：
        - 不按内容分类，统一先尝试自建
        - 自建满载时才溢出到云 API
        """
        # 先尝试所有自建后端
        local_backends = [
            b for b in self.backends.values()
            if b.tier in (ModelTier.LARGE, ModelTier.SMALL)
        ]
        # 按成本排序（便宜的优先）
        local_backends.sort(key=lambda b: b.cost_per_mtok)

        for backend in local_backends:
            if await self.health_check(backend) and backend.pending_count < backend.max_pending:
                return backend

        # 自建都满了，溢出到云 API
        cloud_backends = self._get_backends_by_tier(ModelTier.CLOUD)
        for backend in cloud_backends:
            if await self.health_check(backend):
                logger.info(f"Overflowing to cloud API: {backend.name}")
                return backend

        raise RuntimeError("All backends are unavailable")

    # ----- 请求执行 -----

    async def execute(self, messages: list[dict], stream: bool = False) -> dict:
        """完整的路由 + 调用流程"""
        backend = await self.route(messages)
        backend.pending_count += 1

        try:
            client = await self._get_client()
            payload = {
                "model": backend.model,
                "messages": messages,
                "stream": stream,
            }
            resp = await client.post(
                f"{backend.url}/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(connect=5, read=120, write=10, pool=5),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Backend {backend.name} failed: {e}")
            # 标记为不健康，下次会跳过
            backend.healthy = False
            backend.last_health_check = time.time()
            raise
        finally:
            backend.pending_count = max(0, backend.pending_count - 1)


# ============================================================
# 演示
# ============================================================

async def demo():
    """演示路由逻辑（不实际调用后端）"""
    router = ModelRouter(BACKENDS)

    # 模拟：所有自建后端健康
    for b in router.backends.values():
        b.healthy = True
        b.last_health_check = time.time()

    print("=" * 60)
    print("场景 1：简单问题 → 小模型")
    messages = [{"role": "user", "content": "你好，今天天气怎么样？"}]
    backend = await router.route(messages)
    print(f"  → 路由到: {backend.name} ({backend.tier.value})")

    print("\n场景 2：代码问题 → 大模型")
    messages = [{"role": "user", "content": "帮我写一个快速排序的代码，要求支持泛型"}]
    backend = await router.route(messages)
    print(f"  → 路由到: {backend.name} ({backend.tier.value})")

    print("\n场景 3：长上下文 → 大模型")
    messages = [{"role": "user", "content": "这是一篇很长的文章..." + "内容" * 2000}]
    backend = await router.route(messages)
    print(f"  → 路由到: {backend.name} ({backend.tier.value})")

    print("\n场景 4：大模型满载 → 降级到小模型")
    router.backends["qwen-72b-local"].pending_count = 30  # 满载
    messages = [{"role": "user", "content": "帮我分析这段代码"}]
    backend = await router.route(messages)
    print(f"  → 路由到: {backend.name} ({backend.tier.value})")

    print("\n场景 5：所有自建满载 → 溢出到云 API")
    router.backends["qwen-72b-local"].pending_count = 30
    router.backends["qwen-7b-local"].pending_count = 50
    messages = [{"role": "user", "content": "你好"}]
    backend = await router.route(messages)
    print(f"  → 路由到: {backend.name} ({backend.tier.value})")

    print("\n场景 6：成本感知路由")
    router.backends["qwen-72b-local"].pending_count = 0
    router.backends["qwen-7b-local"].pending_count = 0
    backend = await router.cost_aware_route(messages)
    print(f"  → 路由到: {backend.name} (成本 ¥{backend.cost_per_mtok}/MTok)")

    await router.close()
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
