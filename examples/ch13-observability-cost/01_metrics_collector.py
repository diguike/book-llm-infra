"""
LLM 关键指标采集器 — Prometheus 格式
采集 vLLM 实例的核心指标并暴露给 Prometheus 抓取。

运行：
  pip install -r requirements.txt
  python 01_metrics_collector.py

指标端点：http://localhost:9091/metrics

采集的指标：
  - llm_ttft_seconds:          Time to First Token 分布
  - llm_tps:                   Tokens per Second（单请求）
  - llm_request_duration:      端到端请求耗时
  - llm_tokens_total:          Token 消耗计数
  - llm_backend_pending:       后端排队请求数
  - llm_gpu_cache_usage:       KV Cache 使用率
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
    REGISTRY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metrics_collector")

# ============================================================
# Prometheus 指标定义
# ============================================================

# TTFT 分布（秒）
TTFT_HISTOGRAM = Histogram(
    "llm_ttft_seconds",
    "Time to First Token in seconds",
    labelnames=["model", "backend"],
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# 单请求 TPS
TPS_HISTOGRAM = Histogram(
    "llm_tps",
    "Tokens per second for a single request",
    labelnames=["model"],
    buckets=(5, 10, 20, 30, 50, 80, 100, 150),
)

# 端到端延迟
REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "Total request duration in seconds",
    labelnames=["model", "status"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
)

# Token 计数
TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens processed",
    labelnames=["model", "type"],  # type: prompt / completion
)

# 请求计数
REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total requests",
    labelnames=["model", "status"],
)

# 后端状态（从 vLLM /metrics 抓取）
BACKEND_PENDING = Gauge(
    "llm_backend_pending_requests",
    "Number of pending requests on backend",
    labelnames=["backend"],
)

GPU_CACHE_USAGE = Gauge(
    "llm_gpu_cache_usage_percent",
    "KV Cache usage percentage",
    labelnames=["backend"],
)

BACKEND_RUNNING = Gauge(
    "llm_backend_running_requests",
    "Number of running requests on backend",
    labelnames=["backend"],
)


# ============================================================
# 指标记录器（在 API Gateway 中调用）
# ============================================================

class LLMMetrics:
    """
    在请求处理过程中记录指标。

    使用方式：
        metrics = LLMMetrics()

        # 请求开始
        ctx = metrics.start_request("qwen-7b", "http://backend:8000")

        # 收到第一个 token
        ctx.record_first_token()

        # 请求完成
        ctx.finish(prompt_tokens=100, completion_tokens=50, status="success")
    """

    class RequestContext:
        def __init__(self, model: str, backend: str):
            self.model = model
            self.backend = backend
            self.start_time = time.time()
            self.first_token_time: float | None = None

        def record_first_token(self):
            """记录收到第一个 token 的时间"""
            if self.first_token_time is None:
                self.first_token_time = time.time()
                ttft = self.first_token_time - self.start_time
                TTFT_HISTOGRAM.labels(
                    model=self.model,
                    backend=self.backend,
                ).observe(ttft)
                logger.debug(f"TTFT: {ttft:.3f}s")

        def finish(self, prompt_tokens: int, completion_tokens: int, status: str = "success"):
            """请求完成，记录所有指标"""
            duration = time.time() - self.start_time

            # 请求延迟
            REQUEST_DURATION.labels(model=self.model, status=status).observe(duration)

            # 请求计数
            REQUESTS_TOTAL.labels(model=self.model, status=status).inc()

            # Token 计数
            TOKENS_TOTAL.labels(model=self.model, type="prompt").inc(prompt_tokens)
            TOKENS_TOTAL.labels(model=self.model, type="completion").inc(completion_tokens)

            # TPS（仅 decode 阶段）
            if self.first_token_time and completion_tokens > 0:
                decode_time = time.time() - self.first_token_time
                if decode_time > 0:
                    tps = completion_tokens / decode_time
                    TPS_HISTOGRAM.labels(model=self.model).observe(tps)

            logger.info(
                f"Request finished: model={self.model} "
                f"prompt={prompt_tokens} completion={completion_tokens} "
                f"duration={duration:.2f}s status={status}"
            )

    def start_request(self, model: str, backend: str) -> "LLMMetrics.RequestContext":
        return self.RequestContext(model, backend)


# ============================================================
# 后端指标拉取（定期从 vLLM /metrics 抓取）
# ============================================================

async def scrape_backend_metrics(backends: list[str], interval: int = 15):
    """
    定期从 vLLM 后端拉取指标并更新 Prometheus gauge。
    vLLM 的 /metrics 端点返回 Prometheus 文本格式。
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            for backend_url in backends:
                try:
                    resp = await client.get(f"{backend_url}/metrics")
                    if resp.status_code != 200:
                        continue

                    text = resp.text
                    # 解析几个关键指标
                    for line in text.split("\n"):
                        if line.startswith("#"):
                            continue

                        if "num_requests_waiting" in line and not line.startswith("#"):
                            value = _parse_metric_value(line)
                            if value is not None:
                                BACKEND_PENDING.labels(backend=backend_url).set(value)

                        elif "num_requests_running" in line and not line.startswith("#"):
                            value = _parse_metric_value(line)
                            if value is not None:
                                BACKEND_RUNNING.labels(backend=backend_url).set(value)

                        elif "gpu_cache_usage_perc" in line and not line.startswith("#"):
                            value = _parse_metric_value(line)
                            if value is not None:
                                GPU_CACHE_USAGE.labels(backend=backend_url).set(value * 100)

                except Exception as e:
                    logger.warning(f"Failed to scrape {backend_url}: {e}")

            await asyncio.sleep(interval)


def _parse_metric_value(line: str) -> float | None:
    """从 Prometheus 文本格式中提取数值"""
    try:
        parts = line.strip().split()
        if len(parts) >= 2:
            return float(parts[-1])
    except (ValueError, IndexError):
        pass
    return None


# ============================================================
# Grafana Dashboard JSON（简化版）
# ============================================================

GRAFANA_DASHBOARD_HINT = """
推荐的 Grafana Dashboard 面板：

1. TTFT 分布 (Heatmap)
   Query: histogram_quantile(0.95, rate(llm_ttft_seconds_bucket[5m]))

2. 系统 TPS (Graph)
   Query: rate(llm_tokens_total{type="completion"}[1m])

3. 请求成功率 (Stat)
   Query: sum(rate(llm_requests_total{status="success"}[5m]))
          / sum(rate(llm_requests_total[5m]))

4. 后端队列深度 (Graph)
   Query: llm_backend_pending_requests

5. KV Cache 使用率 (Gauge)
   Query: llm_gpu_cache_usage_percent
"""


# ============================================================
# 主函数
# ============================================================

async def main():
    # 启动 Prometheus metrics 端点
    start_http_server(9091)
    logger.info("Metrics server started on :9091/metrics")

    # 模拟后端列表
    backends = [
        "http://vllm-qwen-7b-0:8000",
        "http://vllm-qwen-7b-1:8000",
    ]

    # 启动后端指标拉取
    scrape_task = asyncio.create_task(scrape_backend_metrics(backends))

    # 演示：模拟一些请求指标
    metrics = LLMMetrics()

    for i in range(5):
        ctx = metrics.start_request("qwen-7b", backends[0])
        await asyncio.sleep(0.1)  # 模拟 prefill
        ctx.record_first_token()
        await asyncio.sleep(0.5)  # 模拟 decode
        ctx.finish(prompt_tokens=100 + i * 50, completion_tokens=50 + i * 20)

    logger.info("Demo metrics recorded. Check http://localhost:9091/metrics")
    logger.info(GRAFANA_DASHBOARD_HINT)

    # 保持运行
    try:
        await scrape_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
