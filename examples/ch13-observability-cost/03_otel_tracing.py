"""
OpenTelemetry LLM 请求追踪示例
为 LLM 推理请求添加全链路追踪，可视化每个阶段的耗时。

运行前提：
  1. 安装依赖：pip install -r requirements.txt
  2. 启动 OTel Collector 或 Jaeger:
     docker run -d -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one:1.62

运行：
  python 03_otel_tracing.py

查看 trace：
  打开 http://localhost:16686 (Jaeger UI)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

# 如果安装了 OTLP exporter，可以发送到 Jaeger/Tempo
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    HAS_OTLP = True
except ImportError:
    HAS_OTLP = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("otel_tracing")

# ============================================================
# OTel 初始化
# ============================================================

def init_tracing(
    service_name: str = "llm-gateway",
    otlp_endpoint: str = "http://localhost:4317",
) -> trace.Tracer:
    """
    初始化 OpenTelemetry tracing。

    Args:
        service_name: 服务名称，在 Jaeger/Grafana 中显示
        otlp_endpoint: OTel Collector / Jaeger 的 gRPC 端点

    Returns:
        配置好的 Tracer 实例
    """
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
        "deployment.environment": "production",
    })

    provider = TracerProvider(resource=resource)

    # Console exporter（开发调试用）
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # OTLP exporter（发送到 Jaeger / Grafana Tempo）
    if HAS_OTLP:
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP exporter configured: {otlp_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to configure OTLP exporter: {e}")

    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)


# ============================================================
# LLM 请求追踪
# ============================================================

class TracedLLMClient:
    """
    带全链路追踪的 LLM 客户端。

    Trace 结构：
      llm.chat_completion (root span)
        ├── auth.verify         (~2ms)
        ├── rate_limit.check    (~3ms)
        ├── model.route         (~1ms)
        ├── backend.request
        │   ├── queue.wait      (可变)
        │   ├── prefill         (跟输入长度正相关)
        │   └── decode          (跟输出长度正相关)
        └── response.stream     (持续到最后一个 token)
    """

    def __init__(self, tracer: trace.Tracer):
        self.tracer = tracer

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
    ) -> dict | AsyncGenerator:
        """
        模拟一个完整的 LLM 请求，包含所有追踪 span。
        """
        with self.tracer.start_as_current_span(
            "llm.chat_completion",
            attributes={
                "llm.model": model,
                "llm.stream": stream,
                "llm.message_count": len(messages),
            },
        ) as root_span:

            # --- 1. 鉴权 ---
            with self.tracer.start_as_current_span("auth.verify") as auth_span:
                await asyncio.sleep(0.002)  # 模拟鉴权延迟
                auth_span.set_attribute("auth.api_key_prefix", "sk-***001")
                auth_span.set_attribute("auth.user", "demo-user")

            # --- 2. 限流检查 ---
            with self.tracer.start_as_current_span("rate_limit.check") as rl_span:
                await asyncio.sleep(0.003)  # 模拟 Redis 查询
                rl_span.set_attribute("rate_limit.rpm_remaining", 55)
                rl_span.set_attribute("rate_limit.tpm_remaining", 95000)

            # --- 3. 模型路由 ---
            with self.tracer.start_as_current_span("model.route") as route_span:
                # 计算输入 token 数（粗估）
                input_text = " ".join(m.get("content", "") for m in messages)
                estimated_input_tokens = len(input_text) // 4
                route_span.set_attribute("route.input_tokens_est", estimated_input_tokens)

                backend_url = "http://vllm-qwen-7b-0:8000"
                route_span.set_attribute("route.backend", backend_url)
                route_span.set_attribute("route.strategy", "least_pending")

            # --- 4. 后端请求 ---
            with self.tracer.start_as_current_span(
                "backend.request",
                attributes={"backend.url": backend_url},
            ) as backend_span:

                # 4a. 队列等待
                with self.tracer.start_as_current_span("queue.wait") as queue_span:
                    queue_time = 0.05  # 模拟 50ms 排队
                    await asyncio.sleep(queue_time)
                    queue_span.set_attribute("queue.wait_ms", queue_time * 1000)
                    queue_span.set_attribute("queue.depth", 3)

                # 4b. Prefill（处理输入）
                with self.tracer.start_as_current_span("prefill") as prefill_span:
                    # Prefill 时间跟输入长度正相关
                    prefill_time = max(0.05, estimated_input_tokens * 0.0002)
                    await asyncio.sleep(prefill_time)
                    prefill_span.set_attribute("prefill.input_tokens", estimated_input_tokens)
                    prefill_span.set_attribute("prefill.duration_ms", prefill_time * 1000)

                # 4c. Decode（生成输出）
                output_tokens = 50  # 模拟生成 50 个 token
                with self.tracer.start_as_current_span("decode") as decode_span:
                    tps = 40  # 模拟 40 tokens/s
                    decode_time = output_tokens / tps
                    await asyncio.sleep(decode_time)
                    decode_span.set_attribute("decode.output_tokens", output_tokens)
                    decode_span.set_attribute("decode.tps", tps)
                    decode_span.set_attribute("decode.duration_ms", decode_time * 1000)

                backend_span.set_attribute("backend.status_code", 200)

            # --- 5. 记录最终指标 ---
            root_span.set_attribute("llm.input_tokens", estimated_input_tokens)
            root_span.set_attribute("llm.output_tokens", output_tokens)
            root_span.set_attribute("llm.total_tokens", estimated_input_tokens + output_tokens)
            root_span.set_attribute(
                "llm.ttft_ms",
                (queue_time + prefill_time) * 1000,
            )

            return {
                "model": model,
                "choices": [{"message": {"role": "assistant", "content": "模拟回复..."}}],
                "usage": {
                    "prompt_tokens": estimated_input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": estimated_input_tokens + output_tokens,
                },
            }


# ============================================================
# 自定义 Span 属性的语义约定
# ============================================================

LLM_SPAN_ATTRIBUTES = """
推荐的 LLM Span 属性命名规范（参考 OpenTelemetry Semantic Conventions for GenAI）：

根 Span:
  gen_ai.system:           "vllm" / "openai" / "anthropic"
  gen_ai.request.model:    请求的模型名
  gen_ai.response.model:   实际响应的模型名
  gen_ai.usage.input_tokens:   输入 token 数
  gen_ai.usage.output_tokens:  输出 token 数

自定义扩展:
  llm.ttft_ms:             Time to First Token (毫秒)
  llm.tps:                 Tokens Per Second
  llm.stream:              是否 streaming
  queue.depth:             队列深度
  queue.wait_ms:           排队等待时间
  prefill.duration_ms:     Prefill 耗时
  decode.duration_ms:      Decode 耗时
  route.backend:           路由到的后端
  route.strategy:          路由策略
"""


# ============================================================
# 演示
# ============================================================

async def main():
    tracer = init_tracing(
        service_name="llm-gateway-demo",
        otlp_endpoint="http://localhost:4317",
    )

    client = TracedLLMClient(tracer)

    print("发送 3 个模拟请求，生成 trace...\n")

    # 请求 1：短输入
    resp = await client.chat_completion(
        model="qwen-7b",
        messages=[{"role": "user", "content": "你好"}],
    )
    print(f"请求 1: {resp['usage']}")

    # 请求 2：中等输入
    resp = await client.chat_completion(
        model="qwen-7b",
        messages=[
            {"role": "system", "content": "你是一个有用的助手。" * 50},
            {"role": "user", "content": "请解释什么是 Transformer 架构？"},
        ],
    )
    print(f"请求 2: {resp['usage']}")

    # 请求 3：长输入
    resp = await client.chat_completion(
        model="qwen-72b",
        messages=[
            {"role": "system", "content": "你是一个代码专家。" * 100},
            {"role": "user", "content": "帮我重构这段代码..." + "x = x + 1\n" * 200},
        ],
    )
    print(f"请求 3: {resp['usage']}")

    print("\n完成！")
    print("如果 Jaeger 在运行，可以在 http://localhost:16686 查看 trace")
    print(f"\n{LLM_SPAN_ATTRIBUTES}")

    # 等待 span export 完成
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
