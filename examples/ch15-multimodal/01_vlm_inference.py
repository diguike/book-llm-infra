"""
Chapter 15 示例 1：用 vLLM 跑 Qwen2.5-VL 多模态推理

演示如何用 vLLM 部署 VLM（Vision-Language Model），处理图片 + 文本的推理请求。
包含单张图片推理、batch 推理、以及性能对比。

依赖：
    pip install vllm Pillow requests

硬件要求：
    - Qwen2.5-VL-7B-Instruct: 至少 1 张 A100 40GB 或 2 张 A10 24GB
    - Qwen2.5-VL-2B-Instruct: 1 张 A10 24GB 即可
"""

import time
from io import BytesIO

import requests
from PIL import Image
from vllm import LLM, SamplingParams


def load_image_from_url(url: str) -> Image.Image:
    """从 URL 下载图片并返回 PIL Image 对象"""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_image_from_file(path: str) -> Image.Image:
    """从本地文件加载图片"""
    return Image.open(path).convert("RGB")


def preprocess_image(image: Image.Image, max_resolution: int = 1344) -> Image.Image:
    """
    预处理图片：限制最大分辨率，保持宽高比。

    生产环境中这一步很重要——一张 4K 图片可能产生上万个 visual tokens，
    直接导致显存爆炸。限制到 1344 是精度和性能的平衡点。
    """
    w, h = image.size
    if max(w, h) <= max_resolution:
        return image

    scale = max_resolution / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def build_vlm_prompt(text: str) -> str:
    """
    构建 Qwen2.5-VL 的 chat prompt 格式。
    <image> placeholder 会被 vLLM 替换为实际的 visual tokens。
    """
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<image>\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def single_image_inference():
    """单张图片推理示例"""
    print("=" * 60)
    print("单张图片推理")
    print("=" * 60)

    # 初始化 vLLM —— 这一步会加载模型权重到 GPU，通常需要 30-60 秒
    # gpu_memory_utilization=0.9 表示允许使用 90% 的显存
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"正在加载模型: {model_name}")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,  # 最大序列长度（包括 visual tokens）
        gpu_memory_utilization=0.9,
        # 如果显存不够，可以设置 tensor_parallel_size=2 做 TP
        # tensor_parallel_size=2,
    )

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

    # 下载一张示例图片
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    print(f"下载图片: {image_url}")
    image = load_image_from_url(image_url)
    image = preprocess_image(image, max_resolution=1344)
    print(f"图片尺寸: {image.size}")

    # 构建请求
    prompt = build_vlm_prompt("请详细描述这张图片的内容。")

    # 推理
    start = time.perf_counter()
    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        sampling_params=sampling_params,
    )
    elapsed = time.perf_counter() - start

    result = outputs[0]
    generated_text = result.outputs[0].text
    num_tokens = len(result.outputs[0].token_ids)

    print(f"\n生成结果: {generated_text}")
    print(f"\n--- 性能指标 ---")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"生成 token 数: {num_tokens}")
    print(f"TTFT (估算): {elapsed - num_tokens * 0.03:.2f}s")  # 粗略估算
    print(f"吞吐: {num_tokens / elapsed:.1f} tok/s")

    return llm


def batch_inference(llm: LLM):
    """
    Batch 推理示例：同时处理多个图片请求。

    vLLM 的 continuous batching 在多模态场景中同样生效。
    但注意：图片请求的 batch 大小通常需要比纯文本小，
    因为每张图片的 visual tokens 会占用大量显存。
    """
    print("\n" + "=" * 60)
    print("Batch 推理 (3 张图片)")
    print("=" * 60)

    sampling_params = SamplingParams(max_tokens=128, temperature=0.7)

    # 准备多个图片请求
    # 实际生产中这些图片来自用户上传
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png",
    ]

    questions = [
        "这张图片里是什么动物？",
        "描述这张图片的内容。",
        "这张照片是在什么场景下拍的？",
    ]

    requests_data = []
    for url, question in zip(image_urls, questions):
        try:
            img = load_image_from_url(url)
            img = preprocess_image(img)
            requests_data.append({
                "prompt": build_vlm_prompt(question),
                "multi_modal_data": {"image": img},
            })
        except Exception as e:
            print(f"跳过图片 {url}: {e}")

    if not requests_data:
        print("没有有效的图片请求")
        return

    # Batch 推理
    start = time.perf_counter()
    outputs = llm.generate(requests_data, sampling_params=sampling_params)
    elapsed = time.perf_counter() - start

    total_tokens = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        n_tokens = len(output.outputs[0].token_ids)
        total_tokens += n_tokens
        print(f"\n请求 {i+1}: {questions[i]}")
        print(f"回答: {text[:200]}...")
        print(f"Token 数: {n_tokens}")

    print(f"\n--- Batch 性能指标 ---")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"请求数: {len(requests_data)}")
    print(f"平均每请求: {elapsed / len(requests_data):.2f}s")
    print(f"总吞吐: {total_tokens / elapsed:.1f} tok/s")


def compare_resolutions(llm: LLM):
    """
    对比不同分辨率下的推理性能。
    这个实验能直观看到分辨率对延迟和显存的影响。
    """
    print("\n" + "=" * 60)
    print("分辨率对性能的影响")
    print("=" * 60)

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    image = load_image_from_url(image_url)
    sampling_params = SamplingParams(max_tokens=64, temperature=0.7)

    resolutions = [224, 448, 896, 1344]

    for res in resolutions:
        resized = preprocess_image(image, max_resolution=res)
        prompt = build_vlm_prompt("What is in this image?")

        start = time.perf_counter()
        outputs = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": resized}}],
            sampling_params=sampling_params,
        )
        elapsed = time.perf_counter() - start

        print(f"分辨率 {res}x{res}: {elapsed:.2f}s, "
              f"图片实际尺寸 {resized.size}")


if __name__ == "__main__":
    # 1. 单张图片推理
    llm = single_image_inference()

    # 2. Batch 推理
    batch_inference(llm)

    # 3. 分辨率对比
    compare_resolutions(llm)
