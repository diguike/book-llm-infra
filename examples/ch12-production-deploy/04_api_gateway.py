"""
LLM API Gateway — 基于 FastAPI
功能：API Key 鉴权、滑动窗口限流（请求级 + Token 级）、负载均衡路由

运行：
  pip install -r requirements.txt
  uvicorn 04_api_gateway:app --host 0.0.0.0 --port 9000

测试：
  curl -X POST http://localhost:9000/v1/chat/completions \
    -H "Authorization: Bearer sk-test-key-001" \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen-7b", "messages": [{"role": "user", "content": "Hello"}]}'
"""

from __future__ import annotations

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_gateway")

# ============================================================
# 配置
# ============================================================

# Redis 用于限流计数
REDIS_URL = "redis://localhost:6379/0"

# 后端 vLLM 实例列表
BACKENDS = {
    "qwen-7b": [
        "http://vllm-qwen-7b-0:8000",
        "http://vllm-qwen-7b-1:8000",
    ],
    "qwen-72b": [
        "http://vllm-qwen-72b-0:8000",
    ],
}

# API Key 配置（生产环境从数据库或 KMS 读取）
API_KEYS = {
    "sk-test-key-001": {
        "name": "测试用户",
        "rpm_limit": 60,            # 每分钟请求数
        "tpm_limit": 100_000,       # 每分钟 token 数
        "allowed_models": ["qwen-7b", "qwen-72b"],
    },
    "sk-prod-key-001": {
        "name": "生产用户",
        "rpm_limit": 300,
        "tpm_limit": 1_000_000,
        "allowed_models": ["qwen-7b", "qwen-72b"],
    },
}


# ============================================================
# 数据模型
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False


# ============================================================
# 鉴权
# ============================================================

def extract_api_key(request: Request) -> str:
    """从 Authorization header 提取 API Key"""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return auth[7:]


async def authenticate(request: Request) -> dict:
    """验证 API Key 并返回用户信息"""
    api_key = extract_api_key(request)
    user_info = API_KEYS.get(api_key)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {**user_info, "api_key": api_key}


# ============================================================
# 限流：滑动窗口
# ============================================================

class RateLimiter:
    """基于 Redis sorted set 的滑动窗口限流器"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def check_rpm(self, api_key: str, limit: int) -> bool:
        """请求级限流：每分钟最多 limit 个请求"""
        now = time.time()
        window_start = now - 60
        key = f"rpm:{api_key}"

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {f"{now}": now})
        pipe.expire(key, 120)
        results = await pipe.execute()

        count = results[1]
        if count >= limit:
            # 超限，移除刚添加的记录
            await self.redis.zrem(key, f"{now}")
            return False
        return True

    async def record_tokens(self, api_key: str, tokens: int):
        """记录已消耗的 token 数"""
        now = time.time()
        key = f"tpm:{api_key}"
        await self.redis.zadd(key, {f"{now}:{tokens}": now})
        await self.redis.expire(key, 120)

    async def check_tpm(self, api_key: str, limit: int) -> bool:
        """Token 级限流：每分钟最多 limit 个 token"""
        now = time.time()
        window_start = now - 60
        key = f"tpm:{api_key}"

        # 清理过期记录
        await self.redis.zremrangebyscore(key, 0, window_start)

        # 统计当前窗口的 token 总数
        members = await self.redis.zrangebyscore(key, window_start, now)
        total_tokens = sum(int(m.split(b":")[1]) for m in members if b":" in m)

        return total_tokens < limit


# ============================================================
# 负载均衡：Least Pending Requests
# ============================================================

class LeastPendingRouter:
    """基于 pending requests 数量的负载均衡"""

    def __init__(self):
        self.pending: dict[str, int] = {}

    def get_backend(self, model: str) -> str:
        """选择 pending 最少的后端"""
        backends = BACKENDS.get(model, [])
        if not backends:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

        # 初始化计数
        for b in backends:
            if b not in self.pending:
                self.pending[b] = 0

        # 选 pending 最少的
        best = min(backends, key=lambda b: self.pending.get(b, 0))
        self.pending[best] += 1
        return best

    def release(self, backend: str):
        """请求完成，释放计数"""
        if backend in self.pending:
            self.pending[backend] = max(0, self.pending[backend] - 1)


# ============================================================
# 应用初始化
# ============================================================

redis_client: Optional[redis.Redis] = None
rate_limiter: Optional[RateLimiter] = None
router = LeastPendingRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, rate_limiter
    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    rate_limiter = RateLimiter(redis_client)
    logger.info("API Gateway started")
    yield
    await redis_client.close()
    logger.info("API Gateway stopped")


app = FastAPI(title="LLM API Gateway", lifespan=lifespan)


# ============================================================
# API 端点
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    user: dict = Depends(authenticate),
):
    api_key = user["api_key"]
    model = body.model

    # 1. 检查模型权限
    if model not in user["allowed_models"]:
        raise HTTPException(status_code=403, detail=f"Model {model} not allowed for this API key")

    # 2. 请求级限流
    if not await rate_limiter.check_rpm(api_key, user["rpm_limit"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded (RPM)")

    # 3. Token 级限流
    if not await rate_limiter.check_tpm(api_key, user["tpm_limit"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded (TPM)")

    # 4. 路由到后端
    backend = router.get_backend(model)
    logger.info(f"Routing {model} request to {backend}, pending: {router.pending.get(backend, 0)}")

    try:
        if body.stream:
            return await _stream_request(backend, body, api_key)
        else:
            return await _normal_request(backend, body, api_key)
    finally:
        router.release(backend)


async def _normal_request(backend: str, body: ChatCompletionRequest, api_key: str) -> dict:
    """非 streaming 请求"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=120, write=10, pool=5)) as client:
        resp = await client.post(
            f"{backend}/v1/chat/completions",
            json=body.model_dump(),
        )
        resp.raise_for_status()
        data = resp.json()

    # 记录 token 消耗（用于 TPM 限流）
    total_tokens = data.get("usage", {}).get("total_tokens", 0)
    if total_tokens > 0:
        await rate_limiter.record_tokens(api_key, total_tokens)

    return data


async def _stream_request(backend: str, body: ChatCompletionRequest, api_key: str):
    """Streaming 请求：SSE 转发"""
    async def event_generator():
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=30, write=10, pool=5)
        ) as client:
            async with client.stream(
                "POST",
                f"{backend}/v1/chat/completions",
                json=body.model_dump(),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        yield f"{line}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx 不缓冲
        },
    )


# ============================================================
# 管理端点
# ============================================================

@app.get("/admin/stats")
async def admin_stats():
    """查看当前路由状态"""
    return {
        "pending_requests": dict(router.pending),
        "backends": BACKENDS,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
