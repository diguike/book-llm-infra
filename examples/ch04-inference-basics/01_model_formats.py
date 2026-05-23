"""
第 4 章 示例 1：加载不同格式模型并对比
对比 PyTorch .bin、SafeTensors、GGUF 格式的加载速度和内存占用。

使用一个小模型 (Qwen/Qwen2-0.5B) 以便在各种环境下都能跑。
"""

import time
import os
import psutil
import torch
from pathlib import Path


def get_memory_mb():
    """获取当前进程的内存占用 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_pytorch_bin(model_name: str = "Qwen/Qwen2-0.5B"):
    """加载 PyTorch .bin 格式"""
    from transformers import AutoModelForCausalLM

    mem_before = get_memory_mb()
    start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    elapsed = time.perf_counter() - start
    mem_after = get_memory_mb()

    print(f"[PyTorch/SafeTensors via HF] 加载时间: {elapsed:.2f}s, "
          f"内存增长: {mem_after - mem_before:.0f} MB")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return elapsed, mem_after - mem_before


def benchmark_safetensors_direct(model_dir: str):
    """直接用 safetensors 库加载（mmap 模式）"""
    from safetensors.torch import load_file
    from glob import glob

    safetensor_files = glob(os.path.join(model_dir, "*.safetensors"))
    if not safetensor_files:
        print("[SafeTensors Direct] 未找到 .safetensors 文件，跳过")
        return None, None

    mem_before = get_memory_mb()
    start = time.perf_counter()

    tensors = {}
    for f in safetensor_files:
        tensors.update(load_file(f, device="cpu"))

    elapsed = time.perf_counter() - start
    mem_after = get_memory_mb()

    total_params = sum(t.numel() for t in tensors.values())
    total_size_mb = sum(t.nbytes for t in tensors.values()) / 1024 / 1024

    print(f"[SafeTensors Direct] 加载时间: {elapsed:.2f}s, "
          f"内存增长: {mem_after - mem_before:.0f} MB, "
          f"参数量: {total_params / 1e6:.1f}M, "
          f"张量大小: {total_size_mb:.0f} MB")

    del tensors
    return elapsed, mem_after - mem_before


def benchmark_gguf(gguf_path: str):
    """加载 GGUF 格式（通过 llama-cpp-python）"""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[GGUF] llama-cpp-python 未安装，跳过。"
              "安装: pip install llama-cpp-python")
        return None, None

    if not os.path.exists(gguf_path):
        print(f"[GGUF] 文件不存在: {gguf_path}，跳过")
        return None, None

    mem_before = get_memory_mb()
    start = time.perf_counter()

    model = Llama(
        model_path=gguf_path,
        n_ctx=512,
        n_gpu_layers=0,  # CPU only
        verbose=False,
    )

    elapsed = time.perf_counter() - start
    mem_after = get_memory_mb()

    print(f"[GGUF] 加载时间: {elapsed:.2f}s, "
          f"内存增长: {mem_after - mem_before:.0f} MB")

    del model
    return elapsed, mem_after - mem_before


def inspect_safetensors_metadata(filepath: str):
    """查看 SafeTensors 文件的元数据（不加载张量数据）"""
    from safetensors import safe_open
    import json

    with safe_open(filepath, framework="pt") as f:
        keys = f.keys()
        print(f"\n文件: {os.path.basename(filepath)}")
        print(f"张量数量: {len(keys)}")
        print(f"前 10 个张量:")
        for i, key in enumerate(list(keys)[:10]):
            tensor = f.get_tensor(key)
            print(f"  {key}: shape={list(tensor.shape)}, dtype={tensor.dtype}")


def main():
    model_name = "Qwen/Qwen2-0.5B"

    print("=" * 60)
    print("模型格式加载对比")
    print(f"模型: {model_name}")
    print("=" * 60)

    # 1. 通过 HuggingFace 加载（自动选择 SafeTensors）
    print("\n--- 1. HuggingFace AutoModel 加载 ---")
    hf_time, hf_mem = benchmark_pytorch_bin(model_name)

    # 2. 直接用 safetensors 库加载
    print("\n--- 2. SafeTensors 直接加载（mmap） ---")
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(model_name)
    st_time, st_mem = benchmark_safetensors_direct(model_dir)

    # 3. 查看 SafeTensors 元数据
    print("\n--- 3. SafeTensors 文件元数据 ---")
    from glob import glob
    safetensor_files = glob(os.path.join(model_dir, "*.safetensors"))
    if safetensor_files:
        inspect_safetensors_metadata(safetensor_files[0])

    # 4. GGUF 加载（需要预先下载 GGUF 文件）
    print("\n--- 4. GGUF 格式加载 ---")
    gguf_path = os.environ.get(
        "GGUF_MODEL_PATH",
        "models/qwen2-0.5b-instruct-q4_k_m.gguf"
    )
    gguf_time, gguf_mem = benchmark_gguf(gguf_path)

    # 汇总
    print("\n" + "=" * 60)
    print("加载对比汇总:")
    print("-" * 60)
    print(f"{'方式':<30} {'时间 (s)':<12} {'内存 (MB)':<12}")
    print("-" * 60)
    if hf_time:
        print(f"{'HuggingFace AutoModel':<30} {hf_time:<12.2f} {hf_mem:<12.0f}")
    if st_time:
        print(f"{'SafeTensors Direct (mmap)':<30} {st_time:<12.2f} {st_mem:<12.0f}")
    if gguf_time:
        print(f"{'GGUF (llama-cpp-python)':<30} {gguf_time:<12.2f} {gguf_mem:<12.0f}")


if __name__ == "__main__":
    main()
