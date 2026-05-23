#!/usr/bin/env python3
"""
对比三种主流 Tokenizer 的分词结果。

用法:
    python 01_tokenizer_compare.py
    python 01_tokenizer_compare.py --text "自定义文本"

依赖:
    pip install tiktoken transformers sentencepiece protobuf
"""

import argparse
import time
import sys


def try_import(module_name: str):
    """尝试导入模块，失败时给出安装提示。"""
    try:
        return __import__(module_name)
    except ImportError:
        print(f"[跳过] {module_name} 未安装，运行: pip install {module_name}")
        return None


def compare_tiktoken(text: str, model: str = "gpt-4o"):
    """使用 tiktoken (OpenAI 的 BPE tokenizer)。"""
    tiktoken = try_import("tiktoken")
    if tiktoken is None:
        return

    print("=" * 60)
    print(f"tiktoken (model: {model})")
    print("=" * 60)

    enc = tiktoken.encoding_for_model(model)

    start = time.perf_counter()
    tokens = enc.encode(text)
    elapsed = time.perf_counter() - start

    # 解码每个 token 查看对应文本
    token_texts = []
    for t in tokens:
        decoded = enc.decode([t])
        token_texts.append(decoded)

    print(f"  词表大小:  {enc.n_vocab}")
    print(f"  Token 数:  {len(tokens)}")
    print(f"  耗时:      {elapsed * 1000:.2f} ms")
    print(f"  Token IDs: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
    print(f"  Token 文本: {token_texts[:20]}{'...' if len(token_texts) > 20 else ''}")
    print()


def compare_huggingface(text: str, model_name: str = "meta-llama/Llama-2-7b-hf"):
    """使用 HuggingFace Transformers 的 tokenizer。"""
    transformers = try_import("transformers")
    if transformers is None:
        return

    print("=" * 60)
    print(f"HuggingFace Transformers (model: {model_name})")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        start = time.perf_counter()
        result = tokenizer.encode(text)
        elapsed = time.perf_counter() - start

        token_texts = [tokenizer.decode([t]) for t in result]

        print(f"  词表大小:  {tokenizer.vocab_size}")
        print(f"  Token 数:  {len(result)}")
        print(f"  耗时:      {elapsed * 1000:.2f} ms")
        print(f"  Token IDs: {result[:20]}{'...' if len(result) > 20 else ''}")
        print(f"  Token 文本: {token_texts[:20]}{'...' if len(token_texts) > 20 else ''}")
    except Exception as e:
        print(f"  [错误] {e}")
        print(f"  提示: 可能需要 huggingface-cli login 或模型需要授权访问")
    print()


def compare_huggingface_qwen(text: str):
    """使用 Qwen2 的 tokenizer 作为中文优化的对比。"""
    transformers = try_import("transformers")
    if transformers is None:
        return

    model_name = "Qwen/Qwen2-0.5B"
    print("=" * 60)
    print(f"HuggingFace Transformers (model: {model_name}) - 中文优化")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        start = time.perf_counter()
        result = tokenizer.encode(text)
        elapsed = time.perf_counter() - start

        token_texts = [tokenizer.decode([t]) for t in result]

        print(f"  词表大小:  {tokenizer.vocab_size}")
        print(f"  Token 数:  {len(result)}")
        print(f"  耗时:      {elapsed * 1000:.2f} ms")
        print(f"  Token IDs: {result[:20]}{'...' if len(result) > 20 else ''}")
        print(f"  Token 文本: {token_texts[:20]}{'...' if len(token_texts) > 20 else ''}")
    except Exception as e:
        print(f"  [错误] {e}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="对比不同 Tokenizer 的分词结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text",
        type=str,
        default="大语言模型的推理优化是一个系统工程问题。KV Cache、FlashAttention、量化都是关键技术。",
        help="要分词的文本",
    )
    parser.add_argument(
        "--tiktoken-model",
        type=str,
        default="gpt-4o",
        help="tiktoken 使用的模型 (default: gpt-4o)",
    )
    args = parser.parse_args()

    print(f"\n输入文本: {args.text}")
    print(f"文本长度: {len(args.text)} 个字符\n")

    # 1. tiktoken (OpenAI)
    compare_tiktoken(args.text, model=args.tiktoken_model)

    # 2. HuggingFace (Llama 2) - 可能需要登录
    compare_huggingface(args.text, model_name="meta-llama/Llama-2-7b-hf")

    # 3. HuggingFace (Qwen2) - 中文优化
    compare_huggingface_qwen(args.text)

    # 总结
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("  - tiktoken (GPT-4o): byte-level BPE，中文效率中等")
    print("  - Llama 2: SentencePiece BPE，中文词表较小，token 数偏多")
    print("  - Qwen2: 扩充中文词表至 15万+，中文 token 效率最高")
    print()
    print("  Token 数直接影响: 推理成本（按 token 计费）、推理速度、context 长度利用率")
    print()


if __name__ == "__main__":
    main()
