"""
Chapter 15 示例 2：用 faster-whisper 构建语音识别 HTTP 服务

使用 faster-whisper（CTranslate2 引擎）部署 Whisper 模型，
通过 FastAPI 提供 OpenAI-compatible 的语音识别 API。

依赖：
    pip install faster-whisper fastapi uvicorn python-multipart

硬件要求：
    - large-v3: 约 4GB 显存（float16）
    - large-v3-turbo: 约 2GB 显存（float16），速度更快
    - medium: 约 2GB 显存（float16），精度略低但速度快
    - 纯 CPU 也能跑，但 large-v3 会很慢（30s 音频约需 15-30s）

启动方式：
    python 02_whisper_server.py
    # 或指定模型和设备
    MODEL_SIZE=large-v3-turbo DEVICE=cuda python 02_whisper_server.py

测试：
    curl -X POST http://localhost:8000/v1/audio/transcriptions \
        -F "file=@audio.wav" \
        -F "language=zh"
"""

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================

MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda")  # "cuda" 或 "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")  # float16, int8, int8_float16
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# 推理参数
DEFAULT_BEAM_SIZE = 5  # beam_size=1 快 30% 但精度下降
DEFAULT_VAD_FILTER = True  # 开启 VAD 过滤，跳过静音段

# ============================================================
# 模型管理
# ============================================================

# 全局模型实例
whisper_model: Optional[WhisperModel] = None


def load_model() -> WhisperModel:
    """
    加载 Whisper 模型。

    faster-whisper 会自动从 Hugging Face 下载 CTranslate2 格式的模型。
    首次加载需要下载，后续使用缓存。

    关于 compute_type 的选择：
    - float16: 速度和精度的最佳平衡，推荐用于 GPU
    - int8_float16: 更省显存，精度损失很小
    - int8: 最省显存，但在某些语言上可能有精度损失
    - float32: 仅用于 CPU，GPU 上没必要
    """
    logger.info(
        f"加载模型: {MODEL_SIZE}, 设备: {DEVICE}, 计算精度: {COMPUTE_TYPE}"
    )
    start = time.perf_counter()

    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE if DEVICE == "cuda" else "float32",
    )

    elapsed = time.perf_counter() - start
    logger.info(f"模型加载完成，耗时 {elapsed:.1f}s")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: 启动时加载模型，关闭时释放"""
    global whisper_model
    whisper_model = load_model()
    # 预热：用一小段静音做一次推理，让 CUDA kernel 编译完成
    warmup(whisper_model)
    yield
    whisper_model = None
    logger.info("模型已释放")


def warmup(model: WhisperModel):
    """
    模型预热。

    CUDA 在首次执行某个 kernel 时需要编译（JIT），会导致第一次请求特别慢。
    预热可以把这个延迟移到启动阶段。
    """
    import numpy as np

    logger.info("正在预热模型...")
    # 生成 1 秒的静音音频（16kHz, float32）
    silence = np.zeros(16000, dtype=np.float32)

    # 写到临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        import wave
        with wave.open(f.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((silence * 32767).astype(np.int16).tobytes())
        temp_path = f.name

    try:
        segments, _ = model.transcribe(temp_path, beam_size=1)
        list(segments)  # 消费 generator
    finally:
        os.unlink(temp_path)

    logger.info("预热完成")


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="Whisper ASR Server",
    description="OpenAI-compatible speech recognition API powered by faster-whisper",
    lifespan=lifespan,
)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(..., description="音频文件（支持 wav, mp3, flac, ogg 等）"),
    language: Optional[str] = Form(None, description="语言代码，如 zh, en, ja"),
    beam_size: int = Form(DEFAULT_BEAM_SIZE, description="Beam search 大小"),
    vad_filter: bool = Form(DEFAULT_VAD_FILTER, description="是否开启 VAD 过滤"),
    word_timestamps: bool = Form(False, description="是否返回逐词时间戳"),
):
    """
    转录音频文件。

    兼容 OpenAI /v1/audio/transcriptions API 格式。
    """
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    # 保存上传的文件到临时目录
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        start = time.perf_counter()

        segments, info = whisper_model.transcribe(
            tmp_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )

        # 收集所有 segment
        result_segments = []
        full_text_parts = []
        for segment in segments:
            seg_data = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            }
            if word_timestamps and segment.words:
                seg_data["words"] = [
                    {
                        "word": w.word,
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "probability": round(w.probability, 3),
                    }
                    for w in segment.words
                ]
            result_segments.append(seg_data)
            full_text_parts.append(segment.text.strip())

        elapsed = time.perf_counter() - start
        audio_duration = info.duration

        # 计算 Real-Time Factor
        rtf = elapsed / audio_duration if audio_duration > 0 else 0

        logger.info(
            f"转录完成: 音频 {audio_duration:.1f}s, "
            f"耗时 {elapsed:.2f}s, RTF={rtf:.3f}, "
            f"语言={info.language} (概率={info.language_probability:.2f})"
        )

        return JSONResponse({
            "text": " ".join(full_text_parts),
            "segments": result_segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(audio_duration, 2),
            "processing_time": round(elapsed, 3),
            "rtf": round(rtf, 4),
        })

    except Exception as e:
        logger.error(f"转录失败: {e}")
        raise HTTPException(status_code=500, detail=f"转录失败: {str(e)}")

    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    """健康检查接口"""
    return {
        "status": "ok" if whisper_model is not None else "loading",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }


@app.get("/v1/models")
async def list_models():
    """列出可用模型（兼容 OpenAI API 格式）"""
    return {
        "object": "list",
        "data": [
            {
                "id": f"whisper-{MODEL_SIZE}",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


if __name__ == "__main__":
    uvicorn.run(
        "02_whisper_server:app",
        host=HOST,
        port=PORT,
        log_level="info",
        # 生产环境建议用 gunicorn + uvicorn worker
        # 但注意：GPU 模型通常只能加载一个实例，所以 worker 数设为 1
        workers=1,
    )
