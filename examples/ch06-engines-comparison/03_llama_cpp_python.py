"""
第 6 章 示例 3：用 llama-cpp-python 做本地推理

llama-cpp-python 是 llama.cpp 的 Python 绑定。
不需要 GPU，CPU 就能跑。适合开发调试和轻量部署。

安装:
  pip install llama-cpp-python

  # 如果需要 GPU 加速 (CUDA):
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

  # Apple Silicon (Metal):
  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

模型下载（GGUF 格式）:
  # 方法 1: 从 Hugging Face 下载
  pip install huggingface_hub
  huggingface-cli download Qwen/Qwen2-0.5B-Instruct-GGUF \
    qwen2-0_5b-instruct-q4_k_m.gguf --local-dir models/

  # 方法 2: 用 Ollama 下载后找到文件
  ollama pull qwen2:0.5b
  # 文件位置: ~/.ollama/models/blobs/
"""

import os
import time
import json
from typing import Optional, Generator


def find_gguf_model() -> Optional[str]:
    """尝试找到本地的 GGUF 模型文件"""
    search_paths = [
        os.environ.get("GGUF_MODEL_PATH", ""),
        "models/qwen2-0_5b-instruct-q4_k_m.gguf",
        "models/qwen2-0.5b-instruct-q4_k_m.gguf",
        os.path.expanduser("~/models/qwen2-0_5b-instruct-q4_k_m.gguf"),
    ]

    for path in search_paths:
        if path and os.path.exists(path):
            return path

    return None


# ============================================================
# 1. 基础推理
# ============================================================

def basic_inference(model_path: str):
    """基础文本生成"""
    from llama_cpp import Llama

    print("=" * 60)
    print("1. 基础推理")
    print("=" * 60)

    # 加载模型
    print(f"加载模型: {model_path}")
    start = time.perf_counter()

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,          # 上下文长度
        n_threads=4,         # CPU 线程数
        n_gpu_layers=0,      # 0 = 纯 CPU; -1 = 全部放 GPU
        verbose=False,
    )

    load_time = time.perf_counter() - start
    print(f"加载耗时: {load_time:.2f}s")

    # 文本补全
    print("\n--- 文本补全 ---")
    start = time.perf_counter()
    output = llm(
        "The capital of France is",
        max_tokens=30,
        temperature=0.0,
        echo=True,  # 包含输入
    )
    elapsed = time.perf_counter() - start

    text = output["choices"][0]["text"]
    usage = output["usage"]
    print(f"  输出: {text}")
    print(f"  Tokens: prompt={usage['prompt_tokens']}, "
          f"completion={usage['completion_tokens']}")
    print(f"  耗时: {elapsed:.2f}s")

    return llm


# ============================================================
# 2. Chat 模式
# ============================================================

def chat_inference(llm):
    """Chat 对话模式"""
    print("\n" + "=" * 60)
    print("2. Chat 对话模式")
    print("=" * 60)

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is Docker? Explain in 2 sentences."},
        ],
        temperature=0.3,
        max_tokens=100,
    )

    reply = output["choices"][0]["message"]["content"]
    print(f"  回复: {reply}")


# ============================================================
# 3. Streaming 输出
# ============================================================

def streaming_chat(llm):
    """Streaming 逐 token 输出"""
    print("\n" + "=" * 60)
    print("3. Streaming 输出")
    print("=" * 60)

    print("  回复: ", end="", flush=True)
    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    stream = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": "Write a short poem about coding."},
        ],
        temperature=0.7,
        max_tokens=100,
        stream=True,
    )

    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")
        if content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            print(content, end="", flush=True)
            token_count += 1

    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) * 1000 if first_token_time else 0
    print(f"\n\n  TTFT: {ttft:.0f}ms")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  Tokens: {token_count}")
    if token_count > 1 and first_token_time:
        tps = token_count / (elapsed - (first_token_time - start))
        print(f"  TPS: {tps:.1f}")


# ============================================================
# 4. JSON 模式
# ============================================================

def json_mode(llm):
    """结构化 JSON 输出"""
    print("\n" + "=" * 60)
    print("4. JSON 模式")
    print("=" * 60)

    from llama_cpp import LlamaGrammar

    # 用 GBNF 语法定义 JSON schema
    # GBNF (GGML BNF) 是 llama.cpp 的语法约束格式
    json_grammar = LlamaGrammar.from_string(r'''
root   ::= "{" ws "\"name\"" ws ":" ws string "," ws "\"age\"" ws ":" ws number "," ws "\"city\"" ws ":" ws string "}" ws
string ::= "\"" [a-zA-Z ]+ "\""
number ::= [0-9]+
ws     ::= [ \t\n]*
''')

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Extract information and respond in JSON."},
            {"role": "user", "content": "Bob is 30 years old and lives in New York."},
        ],
        temperature=0.0,
        max_tokens=100,
        grammar=json_grammar,
    )

    result = output["choices"][0]["message"]["content"]
    print(f"  原始输出: {result}")

    try:
        parsed = json.loads(result)
        print(f"  解析结果: {json.dumps(parsed, indent=4)}")
    except json.JSONDecodeError:
        print("  JSON 解析失败")


# ============================================================
# 5. Embedding
# ============================================================

def embedding_example(model_path: str):
    """文本 Embedding"""
    from llama_cpp import Llama

    print("\n" + "=" * 60)
    print("5. 文本 Embedding")
    print("=" * 60)

    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        embedding=True,  # 开启 embedding 模式
        verbose=False,
    )

    texts = [
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "I love eating pizza.",
    ]

    embeddings = []
    for text in texts:
        emb = llm.embed(text)
        embeddings.append(emb)
        print(f"  \"{text[:40]}...\" → dim={len(emb)}")

    # 计算相似度
    import math

    def cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

    print(f"\n  相似度:")
    print(f"  ML vs DL:    {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"  ML vs Pizza: {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
    print(f"  DL vs Pizza: {cosine_similarity(embeddings[1], embeddings[2]):.4f}")


# ============================================================
# 6. 性能测试
# ============================================================

def performance_test(llm):
    """简单的性能测试"""
    print("\n" + "=" * 60)
    print("6. 性能测试")
    print("=" * 60)

    prompts = [
        "What is HTTP?",
        "Explain TCP in one sentence.",
        "What is a database?",
        "Define API.",
        "What is JSON?",
    ]

    total_tokens = 0
    start = time.perf_counter()

    for prompt in prompts:
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        total_tokens += output["usage"]["completion_tokens"]

    elapsed = time.perf_counter() - start

    print(f"  请求数: {len(prompts)}")
    print(f"  总 token: {total_tokens}")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  吞吐量: {total_tokens / elapsed:.1f} tokens/s")
    print(f"  (注意: 这是顺序执行, llama.cpp 单请求场景)")


def main():
    model_path = find_gguf_model()

    if model_path is None:
        print("未找到 GGUF 模型文件")
        print("\n下载方法:")
        print("  pip install huggingface_hub")
        print("  mkdir -p models")
        print("  huggingface-cli download Qwen/Qwen2-0.5B-Instruct-GGUF \\")
        print("    qwen2-0_5b-instruct-q4_k_m.gguf --local-dir models/")
        print("\n或设置环境变量:")
        print("  export GGUF_MODEL_PATH=/path/to/model.gguf")
        return

    print(f"GGUF 模型: {model_path}")
    print(f"文件大小: {os.path.getsize(model_path) / 1e6:.1f} MB")

    try:
        from llama_cpp import Llama
    except ImportError:
        print("需要安装 llama-cpp-python:")
        print("  pip install llama-cpp-python")
        return

    llm = basic_inference(model_path)
    chat_inference(llm)
    streaming_chat(llm)
    json_mode(llm)
    performance_test(llm)

    # Embedding 需要重新加载（不同模式）
    embedding_example(model_path)


if __name__ == "__main__":
    main()
