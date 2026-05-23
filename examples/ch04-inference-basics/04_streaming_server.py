"""
第 4 章 示例 4：用 FastAPI 实现一个最简 Streaming LLM Server

实现 OpenAI 兼容的 /v1/chat/completions 接口，支持 streaming。
可以用 OpenAI SDK 直接调用。

启动: uvicorn 04_streaming_server:app --host 0.0.0.0 --port 8000
测试: python 04_streaming_server.py --test
"""

import asyncio
import json
import time
import uuid
import argparse
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ============================================================
# 数据模型（OpenAI 兼容格式）
# ============================================================

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[list[str]] = None


# ============================================================
# 模型推理（懒加载）
# ============================================================

_model = None
_tokenizer = None


def get_model():
    """懒加载模型"""
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2-0.5B-Instruct"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"加载模型 {model_name} 到 {device}...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        _model.eval()
        print("模型加载完成")

    return _model, _tokenizer


async def generate_tokens(
    messages: list[Message],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """
    逐 token 生成，yield 每个 token 的文本。
    在真实场景中，这里会调用 vLLM / SGLang 等推理引擎。
    """
    import torch

    model, tokenizer = get_model()
    device = next(model.parameters()).device

    # 构造 chat prompt
    chat_text = ""
    for msg in messages:
        if msg.role == "system":
            chat_text += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif msg.role == "user":
            chat_text += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            chat_text += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    chat_text += "<|im_start|>assistant\n"

    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        # Prefill
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        for i in range(max_tokens):
            token_id = next_token.item()

            # 检查停止条件
            if token_id == tokenizer.eos_token_id:
                break

            # Decode token 为文本
            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
            )

            # 跳过特殊 token 标记
            if token_text.startswith("<|") and token_text.endswith("|>"):
                break

            yield token_text

            # Decode step
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # 让出控制权，允许其他请求处理
            await asyncio.sleep(0)


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(title="Mini LLM Server", version="0.1.0")


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [{
            "id": "local-model",
            "object": "model",
            "owned_by": "local",
        }]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI 兼容的 Chat Completions 接口。
    支持 stream=true 和 stream=false。
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            stream_response(request, request_id, created),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
            },
        )
    else:
        return await non_stream_response(request, request_id, created)


async def stream_response(
    request: ChatCompletionRequest,
    request_id: str,
    created: int,
) -> AsyncGenerator[str, None]:
    """
    SSE Streaming 响应。
    每个 token 发送一个 SSE event。
    """
    # 发送 role 信息
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }]
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # 逐 token 发送
    async for token_text in generate_tokens(
        request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    ):
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": token_text},
                "finish_reason": None,
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # 发送结束标记
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }]
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def non_stream_response(
    request: ChatCompletionRequest,
    request_id: str,
    created: int,
) -> dict:
    """非 Streaming 响应：收集所有 token 后一次性返回"""
    content_parts = []
    async for token_text in generate_tokens(
        request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    ):
        content_parts.append(token_text)

    content = "".join(content_parts)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": -1,  # 简化，不计算
            "completion_tokens": len(content_parts),
            "total_tokens": -1,
        }
    }


# ============================================================
# 测试客户端
# ============================================================

def test_streaming():
    """用 OpenAI SDK 测试 streaming 接口"""
    try:
        from openai import OpenAI
    except ImportError:
        print("需要安装 openai: pip install openai")
        return

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    print("=== Streaming 测试 ===")
    print("请求: 'What is 2+2?'")
    print("响应: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 2+2? Answer in one sentence."},
        ],
        max_tokens=50,
        temperature=0.0,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print("\n")

    print("=== 非 Streaming 测试 ===")
    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "user", "content": "Say hello in 3 words."},
        ],
        max_tokens=20,
        temperature=0.0,
        stream=False,
    )
    print(f"响应: {response.choices[0].message.content}")


def test_with_curl():
    """打印 curl 测试命令"""
    print("\n用 curl 测试 streaming:")
    print("""
curl -N http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "stream": true
  }'
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="运行测试客户端")
    args = parser.parse_args()

    if args.test:
        test_streaming()
        test_with_curl()
    else:
        import uvicorn
        print("启动 Mini LLM Server...")
        print("测试: python 04_streaming_server.py --test")
        uvicorn.run(app, host="0.0.0.0", port=8000)
