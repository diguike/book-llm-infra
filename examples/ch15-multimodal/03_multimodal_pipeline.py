"""
Chapter 15 示例 3：多模态处理 Pipeline

演示一个完整的多模态 Agent 输入处理 pipeline：
- 接收包含文本、图片、音频的混合请求
- 各模态并行预处理
- 合并后送入 VLM 做推理

这是一个架构示例，展示生产环境中多模态 pipeline 的设计模式。

依赖：
    pip install Pillow aiohttp aiofiles faster-whisper

注意：
    这个示例侧重架构设计，VLM 推理部分用 mock 替代。
    实际使用时替换为 vLLM 的调用即可（参考 01_vlm_inference.py）。
"""

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("multimodal-pipeline")


# ============================================================
# 数据结构定义
# ============================================================


class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class ModalityInput:
    """单个模态的原始输入"""
    type: ModalityType
    data: bytes | str  # bytes: 二进制数据, str: 文本或 URL
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessedModality:
    """预处理后的模态数据"""
    type: ModalityType
    tokens: Optional[list] = None  # tokenized 文本
    image: Optional[Image.Image] = None  # 处理后的图片
    text: Optional[str] = None  # 转录/提取的文本
    estimated_tokens: int = 0  # 预估的 token 数量
    processing_time_ms: float = 0


@dataclass
class MultimodalRequest:
    """一个完整的多模态请求"""
    request_id: str
    modalities: list[ModalityInput]
    system_prompt: str = "You are a helpful assistant."
    max_tokens: int = 256
    temperature: float = 0.7


@dataclass
class PipelineResult:
    """Pipeline 处理结果"""
    request_id: str
    generated_text: str
    total_input_tokens: int
    processing_times: dict  # 各阶段耗时
    total_time_ms: float


# ============================================================
# 预处理器
# ============================================================


class ImageProcessor:
    """
    图片预处理器。

    负责：
    1. 解码各种格式的图片
    2. Resize 到合适的分辨率（限制 visual tokens 数量）
    3. 估算 visual token 数量
    """

    def __init__(self, max_resolution: int = 1344, patch_size: int = 14):
        self.max_resolution = max_resolution
        self.patch_size = patch_size

    def estimate_visual_tokens(self, width: int, height: int) -> int:
        """
        估算图片会产生多少 visual tokens。

        大多数 VLM 使用 ViT，把图片切成 patch_size x patch_size 的 patch。
        每个 patch 对应一个 token。
        """
        n_patches_w = width // self.patch_size
        n_patches_h = height // self.patch_size
        return n_patches_w * n_patches_h

    async def process(self, data: bytes | str) -> ProcessedModality:
        start = time.perf_counter()

        # 解码图片
        if isinstance(data, bytes):
            image = Image.open(io.BytesIO(data)).convert("RGB")
        elif isinstance(data, str) and data.startswith("data:image"):
            # Base64 编码的图片
            b64_data = data.split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
        elif isinstance(data, str):
            # 假设是文件路径
            image = Image.open(data).convert("RGB")
        else:
            raise ValueError(f"不支持的图片数据类型: {type(data)}")

        original_size = image.size
        logger.info(f"原始图片尺寸: {original_size}")

        # Resize
        w, h = image.size
        if max(w, h) > self.max_resolution:
            scale = self.max_resolution / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"Resize: {original_size} -> {image.size}")

        estimated_tokens = self.estimate_visual_tokens(*image.size)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"图片处理完成: {image.size}, "
            f"预估 {estimated_tokens} tokens, "
            f"耗时 {elapsed_ms:.1f}ms"
        )

        return ProcessedModality(
            type=ModalityType.IMAGE,
            image=image,
            estimated_tokens=estimated_tokens,
            processing_time_ms=elapsed_ms,
        )


class AudioProcessor:
    """
    音频预处理器。

    负责：
    1. 用 Whisper 做语音识别（STT）
    2. 返回转录文本
    3. 估算 token 数量

    生产环境中，这一步的延迟最大（数百毫秒到几秒），
    所以一定要和图片处理并行执行。
    """

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        初始化时使用较小的模型做示例。
        生产环境建议用 large-v3 或 large-v3-turbo。
        """
        self.model_size = model_size
        self.device = device
        self._model = None

    def _get_model(self):
        """懒加载模型——只在第一次使用时加载"""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="float32" if self.device == "cpu" else "float16",
                )
                logger.info(f"Whisper 模型已加载: {self.model_size}")
            except ImportError:
                logger.warning(
                    "faster-whisper 未安装，音频处理将使用 mock 模式"
                )
        return self._model

    async def process(self, data: bytes | str) -> ProcessedModality:
        start = time.perf_counter()

        model = self._get_model()

        if model is None:
            # Mock 模式：模拟转录结果
            await asyncio.sleep(0.5)  # 模拟处理延迟
            text = "[Mock transcription: 这是一段语音的转录结果]"
        else:
            # 实际转录
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                if isinstance(data, bytes):
                    tmp.write(data)
                elif isinstance(data, str):
                    # 文件路径
                    with open(data, "rb") as f:
                        tmp.write(f.read())
                tmp.flush()

                segments, info = model.transcribe(
                    tmp.name,
                    beam_size=5,
                    vad_filter=True,
                )
                text = " ".join(seg.text.strip() for seg in segments)

        # 粗略估算 token 数：中文约 1.5 token/字，英文约 1.3 token/词
        estimated_tokens = len(text) * 2  # 保守估计

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"音频处理完成: 转录文本长度 {len(text)} 字, "
            f"预估 {estimated_tokens} tokens, "
            f"耗时 {elapsed_ms:.1f}ms"
        )

        return ProcessedModality(
            type=ModalityType.AUDIO,
            text=text,
            estimated_tokens=estimated_tokens,
            processing_time_ms=elapsed_ms,
        )


class TextProcessor:
    """
    文本预处理器。最简单的模态——基本上就是 tokenize。
    """

    async def process(self, data: str) -> ProcessedModality:
        start = time.perf_counter()

        # 粗略估算 token 数
        estimated_tokens = len(data) * 2  # 保守估计

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ProcessedModality(
            type=ModalityType.TEXT,
            text=data,
            estimated_tokens=estimated_tokens,
            processing_time_ms=elapsed_ms,
        )


# ============================================================
# Pipeline 核心
# ============================================================


class MultimodalPipeline:
    """
    多模态处理 Pipeline。

    核心设计原则：
    1. 各模态预处理并行执行
    2. 预处理完成后合并为统一格式
    3. 有显存预算控制——如果 visual tokens 过多，降低分辨率
    4. 超时控制——如果某个模态处理太慢，使用 fallback
    """

    def __init__(
        self,
        max_total_tokens: int = 8192,
        max_visual_tokens: int = 4096,
        preprocessing_timeout: float = 10.0,  # 秒
    ):
        self.max_total_tokens = max_total_tokens
        self.max_visual_tokens = max_visual_tokens
        self.preprocessing_timeout = preprocessing_timeout

        # 初始化各模态处理器
        self.image_processor = ImageProcessor(max_resolution=1344)
        self.audio_processor = AudioProcessor(model_size="base", device="cpu")
        self.text_processor = TextProcessor()

        # 处理器映射
        self._processors = {
            ModalityType.TEXT: self.text_processor.process,
            ModalityType.IMAGE: self.image_processor.process,
            ModalityType.AUDIO: self.audio_processor.process,
        }

    async def preprocess(
        self, modalities: list[ModalityInput]
    ) -> list[ProcessedModality]:
        """
        并行预处理所有模态。

        关键点：图片和音频的处理是完全独立的，必须并行执行。
        串行处理是常见的性能错误。
        """
        tasks = []
        for mod in modalities:
            processor = self._processors.get(mod.type)
            if processor is None:
                logger.warning(f"不支持的模态类型: {mod.type}")
                continue
            tasks.append(processor(mod.data))

        # 并行执行，带超时控制
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.preprocessing_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"预处理超时 ({self.preprocessing_timeout}s)，"
                "部分模态可能未处理完成"
            )
            # 返回已完成的结果
            results = []
            for task in tasks:
                if task.done():
                    results.append(task.result())

        # 过滤掉异常
        processed = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"预处理失败: {r}")
            else:
                processed.append(r)

        return processed

    def check_token_budget(
        self, processed: list[ProcessedModality]
    ) -> list[ProcessedModality]:
        """
        检查 token 预算。

        如果 visual tokens 超过上限，需要降级处理：
        - 降低图片分辨率
        - 减少图片数量
        - 截断文本
        """
        total_visual_tokens = sum(
            p.estimated_tokens for p in processed if p.type == ModalityType.IMAGE
        )

        if total_visual_tokens > self.max_visual_tokens:
            logger.warning(
                f"Visual tokens ({total_visual_tokens}) 超过预算 "
                f"({self.max_visual_tokens})，降低图片分辨率"
            )
            # 按比例降低所有图片的分辨率
            scale = (self.max_visual_tokens / total_visual_tokens) ** 0.5
            for p in processed:
                if p.type == ModalityType.IMAGE and p.image is not None:
                    w, h = p.image.size
                    new_w = max(224, int(w * scale))
                    new_h = max(224, int(h * scale))
                    p.image = p.image.resize((new_w, new_h), Image.LANCZOS)
                    p.estimated_tokens = self.image_processor.estimate_visual_tokens(
                        new_w, new_h
                    )
                    logger.info(f"图片降级: ({w},{h}) -> ({new_w},{new_h})")

        total_tokens = sum(p.estimated_tokens for p in processed)
        logger.info(
            f"Token 预算检查: 总计 {total_tokens} tokens "
            f"(上限 {self.max_total_tokens})"
        )

        return processed

    def build_prompt(
        self,
        processed: list[ProcessedModality],
        system_prompt: str,
    ) -> dict:
        """
        将处理后的各模态数据合并为 VLM 可接受的 prompt 格式。

        返回的 dict 可以直接传给 vLLM 的 generate() 方法。
        """
        text_parts = []
        images = []

        for p in processed:
            if p.type == ModalityType.TEXT:
                text_parts.append(p.text)
            elif p.type == ModalityType.IMAGE:
                text_parts.append("<image>")
                images.append(p.image)
            elif p.type == ModalityType.AUDIO:
                # 音频已转为文本
                text_parts.append(f"[用户语音转录] {p.text}")

        user_content = "\n".join(text_parts)

        # 构建 chat prompt（Qwen2.5-VL 格式）
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        result = {"prompt": prompt}
        if images:
            result["multi_modal_data"] = {"image": images[0] if len(images) == 1 else images}

        return result

    async def process_request(self, request: MultimodalRequest) -> PipelineResult:
        """
        处理一个完整的多模态请求。

        流程：预处理 → Token 预算检查 → 构建 Prompt → 推理
        """
        pipeline_start = time.perf_counter()
        timing = {}

        # Step 1: 并行预处理
        logger.info(
            f"[{request.request_id}] 开始处理，"
            f"包含 {len(request.modalities)} 个模态输入"
        )

        preprocess_start = time.perf_counter()
        processed = await self.preprocess(request.modalities)
        timing["preprocess_ms"] = (time.perf_counter() - preprocess_start) * 1000

        # 记录各模态的处理时间
        for p in processed:
            timing[f"{p.type.value}_ms"] = p.processing_time_ms

        # Step 2: Token 预算检查
        processed = self.check_token_budget(processed)
        total_input_tokens = sum(p.estimated_tokens for p in processed)

        # Step 3: 构建 Prompt
        prompt_data = self.build_prompt(processed, request.system_prompt)

        # Step 4: 推理（这里用 mock，实际使用 vLLM）
        inference_start = time.perf_counter()
        generated_text = await self._mock_inference(prompt_data, request)
        timing["inference_ms"] = (time.perf_counter() - inference_start) * 1000

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        timing["total_ms"] = total_ms

        logger.info(
            f"[{request.request_id}] 处理完成，"
            f"总耗时 {total_ms:.0f}ms，"
            f"输入 {total_input_tokens} tokens"
        )

        return PipelineResult(
            request_id=request.request_id,
            generated_text=generated_text,
            total_input_tokens=total_input_tokens,
            processing_times=timing,
            total_time_ms=total_ms,
        )

    async def _mock_inference(
        self, prompt_data: dict, request: MultimodalRequest
    ) -> str:
        """
        Mock 推理。

        实际使用时替换为：
            outputs = llm.generate([prompt_data], sampling_params)
            return outputs[0].outputs[0].text

        参考 01_vlm_inference.py
        """
        await asyncio.sleep(0.2)  # 模拟推理延迟
        has_image = "multi_modal_data" in prompt_data
        return (
            f"[Mock 推理结果] 收到{'图文' if has_image else '纯文本'}请求，"
            f"输入 prompt 长度: {len(prompt_data['prompt'])} 字符"
        )


# ============================================================
# 使用示例
# ============================================================


async def demo_text_only():
    """纯文本请求"""
    print("\n" + "=" * 60)
    print("Demo 1: 纯文本请求")
    print("=" * 60)

    pipeline = MultimodalPipeline()

    request = MultimodalRequest(
        request_id="req-001",
        modalities=[
            ModalityInput(
                type=ModalityType.TEXT,
                data="帮我写一首关于编程的诗。",
            ),
        ],
    )

    result = await pipeline.process_request(request)
    print(f"结果: {result.generated_text}")
    print(f"耗时: {result.processing_times}")


async def demo_image_and_text():
    """图片 + 文本请求"""
    print("\n" + "=" * 60)
    print("Demo 2: 图片 + 文本请求")
    print("=" * 60)

    pipeline = MultimodalPipeline()

    # 创建一张测试图片
    test_image = Image.new("RGB", (2048, 1536), color=(100, 150, 200))
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    request = MultimodalRequest(
        request_id="req-002",
        modalities=[
            ModalityInput(type=ModalityType.IMAGE, data=img_bytes),
            ModalityInput(
                type=ModalityType.TEXT,
                data="这张图片里有什么？请详细描述。",
            ),
        ],
    )

    result = await pipeline.process_request(request)
    print(f"结果: {result.generated_text}")
    print(f"输入 tokens: {result.total_input_tokens}")
    print(f"耗时明细: {result.processing_times}")


async def demo_multimodal():
    """图片 + 音频 + 文本请求"""
    print("\n" + "=" * 60)
    print("Demo 3: 多模态请求 (图片 + 音频 + 文本)")
    print("=" * 60)

    pipeline = MultimodalPipeline()

    # 创建测试数据
    test_image = Image.new("RGB", (1024, 768), color=(200, 100, 50))
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    request = MultimodalRequest(
        request_id="req-003",
        modalities=[
            ModalityInput(type=ModalityType.IMAGE, data=img_bytes),
            ModalityInput(
                type=ModalityType.AUDIO,
                data=b"fake-audio-data",  # Mock 音频数据
            ),
            ModalityInput(
                type=ModalityType.TEXT,
                data="请结合图片和语音内容给出建议。",
            ),
        ],
    )

    result = await pipeline.process_request(request)
    print(f"结果: {result.generated_text}")
    print(f"输入 tokens: {result.total_input_tokens}")
    print(f"耗时明细:")
    for k, v in result.processing_times.items():
        print(f"  {k}: {v:.1f}ms")


async def demo_token_budget():
    """演示 token 预算控制"""
    print("\n" + "=" * 60)
    print("Demo 4: Token 预算控制 (多张大图)")
    print("=" * 60)

    # 设置较低的 visual token 预算
    pipeline = MultimodalPipeline(max_visual_tokens=2000)

    # 创建多张大图
    modalities = []
    for i in range(3):
        img = Image.new("RGB", (2048, 2048), color=(i * 80, 100, 200))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        modalities.append(ModalityInput(type=ModalityType.IMAGE, data=img_bytes.getvalue()))

    modalities.append(
        ModalityInput(type=ModalityType.TEXT, data="对比这三张图片的差异。")
    )

    request = MultimodalRequest(request_id="req-004", modalities=modalities)

    result = await pipeline.process_request(request)
    print(f"最终输入 tokens: {result.total_input_tokens}")
    print(f"（预算 2000 visual tokens，超出时会自动降低分辨率）")


async def main():
    await demo_text_only()
    await demo_image_and_text()
    await demo_multimodal()
    await demo_token_budget()


if __name__ == "__main__":
    asyncio.run(main())
