# 第 15 章 多模态模型的基础设施

前面十四章，所有输入都是文本。但现实世界不只有文字——用户会发图片问"这是什么虫子"，会丢一段语音让 Agent 帮忙翻译，甚至会传一段视频让模型做摘要。2025 年以来，多模态能力已经从"锦上添花"变成了"生产刚需"。

本章聚焦多模态带来的 Infra 变化：当模型不再只吃文本，推理延迟、显存管理和限流策略都需要对应调整。

## 15.1 Vision-Language 模型的推理

### VLM 的基本架构

Vision-Language Model（VLM，视觉-语言模型，能同时理解图片和文本的多模态模型，也常被称为 MLLM / Multimodal LLM）的核心思路很直接：把一个 Vision Encoder（视觉编码器，把图片转成向量序列的神经网络模块）和一个 LLM 拼在一起。图片经过 Vision Encoder（通常是 ViT 变体）变成一组 visual tokens（视觉 token，图片被编码后产生的向量序列，作用类似文本 tokenizer 的输出，只是每个 token 代表一块图像区域），然后和文本 tokens 一起送进 LLM 做自回归生成。

典型的架构长这样：

```
Image → Vision Encoder (ViT) → Projection Layer → Visual Tokens
                                                        ↓
Text Prompt → Tokenizer → Text Tokens → [Visual Tokens + Text Tokens] → LLM → Output
```

图里有两个关键组件需要展开：

- **Vision Encoder 一般是 ViT**（Vision Transformer，把图片切成小块再用 Transformer 编码的视觉模型，参考论文 [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)）。它把图片切成固定大小的 patch（小图块，把整张图均匀切分得到的小方块，每块作为一个 token 进入 Transformer，比如 14×14 像素一个 patch），每个 patch 用线性投影变成一个向量，再走标准的 Transformer 编码——和 LLM 处理文本的方式如出一辙，只不过 token 变成了图像 patch。
- **Projection Layer（投影层，把一种向量空间映射到另一种的简单网络）是一个线性层或小 MLP**（Multi-Layer Perceptron，多层感知机，最基础的全连接神经网络结构，几层 Linear + 激活函数堆起来）。它的作用是尺寸适配：ViT 输出向量维度（比如 1024）和 LLM 的 embedding 维度（比如 4096）通常不一致，Projection Layer 把 ViT 输出投影到 LLM 能直接消费的维度。这一层很薄，但训练时通常只训它（冻住 ViT 和 LLM），因为它承担着「桥接两个模态空间」的核心职责。

不同的模型在"怎么把图片信息塞进 LLM"这件事上各有各的做法：

- **LLaVA**（Large Language and Vision Assistant，最早把 CLIP 视觉编码器 + LLM 用简单投影层拼起来的开源 VLM 系列，社区基线）：用一个简单的 MLP 做 projection，把 ViT 输出投影到 LLM 的 embedding 空间。优点是架构简洁，训练成本低。
- **Qwen-VL / Qwen2.5-VL**（阿里通义千问的视觉-语言模型系列）：阿里的方案，支持动态分辨率输入。Qwen2.5-VL 在 A100 上用 vLLM 跑图片推理能做到约 20.89 req/s（concurrency=50, 7B 模型），视频则降到 7.35 req/s。
- **InternVL 系列**（上海 AI Lab 出品的开源 VLM 系列，对标 GPT-4V，注重榜单表现）：InternVL3.5 在多个 benchmark（评测集，用一组固定题目和指标横向比较模型能力，类比前端的 Lighthouse 跑分）上表现强劲，例如 MMVet（综合能力评测，考察文字识别、推理、空间关系等）76.6、MMStar（针对 VLM 的多任务评测）65.0。小模型 InternVL3.5-2B 在吞吐和延迟上表现出色，适合对性能敏感的场景。

### 图片预处理：分辨率和 patch 数量

VLM 推理中，一个经常被忽视但影响巨大的因素是图片预处理。

ViT 会把图片切成固定大小的 patch（通常是 14×14 或 16×16 像素），每个 patch 变成一个 token。一张 224×224 的图片会产生 256 个 visual tokens，而 448×448 就变成 1024 个。

这意味着什么？

- **分辨率翻倍，token 数量翻四倍**。一张 1344×1344 的高分辨率图片可能产生 9216 个 visual tokens，比一整段文本 prompt 还长。
- **显存占用和计算量随 token 数二次增长**（因为 attention 是 O(n²)）。
- 实际生产中，你会看到图片请求的 TTFT（Time To First Token，从请求进来到吐出第一个 token 的延迟，是衡量交互体验的关键指标）比纯文本请求高 3-10 倍。

动态分辨率（dynamic resolution，根据原图尺寸动态决定切分方式，而不是统一缩到固定大小）是当前的主流方案。Qwen2.5-VL 支持将图片按原始比例切分成多个 tile（图块，把一张大图均匀切成几块小图分别送入编码器，类比前端切瓦片地图），每个 tile 独立编码，避免了强制 resize 导致的信息损失。但 tile 越多，token 越多，计算越贵。业界还有 NaViT（Native Resolution ViT，支持原生任意分辨率打包的 ViT 变体）和 AnyRes（LLaVA 系列引入的多分辨率切图策略）等更进一步的方案。

**实际建议**：在生产环境中，对用户上传的图片做预处理——限制最大分辨率（比如长边不超过 1344），在清晰度和推理成本之间找平衡。

### vLLM 对 VLM 的支持

vLLM 从 2024 年底开始逐步支持多模态模型，到 2025 年的 V1 版本已经相当成熟。几个关键能力：

1. **Encoder Cache**（编码器缓存）：Visual encoder 的输出会缓存在 GPU 上。如果同一张图片在多个请求中出现（比如同一张图的不同问题），不需要重复编码。
2. **Encoder-aware Scheduler**（编码器感知调度器）：调度器知道每个请求里多模态 embedding 的位置，能更好地管理显存。
3. **支持的模型列表不断扩大**：LLaVA、Qwen-VL、InternVL、Phi-3-Vision（微软 Phi-3 系列的视觉版本，小参数量、可在消费级 GPU 跑）、Pixtral（Mistral AI 推出的 12B 开源 VLM）等主流 VLM 都已支持。

用 vLLM 跑 VLM 推理的基本方式：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

# 图片通过 multi_modal_data 传入
outputs = llm.generate(
    [{
        "prompt": "<|im_start|>user\n<image>\n这张图片里有什么？<|im_end|>\n<|im_start|>assistant\n",
        "multi_modal_data": {"image": image},
    }],
    sampling_params=SamplingParams(max_tokens=256, temperature=0.7),
)
```

vLLM 从 v0.6.x 开始持续扩展多模态支持，输入侧（图片、视频、音频）已经相当稳定。输出侧的全模态能力（直接生成图片或音频）仍在开发中，相关进展可关注 vLLM 的 GitHub roadmap（[github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)）和发版日志，不同 minor 版本的多模态能力差距不小，升级前最好对照官方变更说明确认。

### 与纯文本 LLM 推理的性能差异

来看一组实际数据（A100 40GB, vLLM, concurrency=50）：

| 模型 | 图片 req/s | 视频 req/s | 纯文本 req/s（估算） |
|------|-----------|-----------|-------------------|
| Qwen2.5-VL-7B | 20.89 | 7.35 | ~45-60 |
| InternVL3.5-4B | 更高 | - | ~70-90 |
| InternVL3.5-2B | 最高 | - | ~100+ |

从数据看，VLM 和纯文本 LLM 的吞吐差距集中在三个方向：

1. **视频推理吞吐大幅低于图片**，因为视频会采样多帧，每帧都要编码。
2. **TTFT 显著增加**：纯文本的 TTFT 通常在 50-200ms，VLM 的图片请求可能到 500ms-2s。
3. **显存使用更不可预测**：文本请求的 token 数相对稳定，但一张高分辨率图片可能突然吃掉几个 GB 的显存。

这对 Infra 意味着什么？你的 auto-scaler 和请求限流策略需要考虑模态差异。不能简单地用"每秒请求数"来限流，得按"每秒 token 数"或"每秒计算量"来算。

## 15.2 语音模型的服务化

### Whisper：语音识别的工业标准

OpenAI 的 Whisper（OpenAI 开源的 ASR / Automatic Speech Recognition 模型，即自动语音识别，把语音音频转成文字）模型几乎统治了开源语音识别领域。Whisper large-v3 有 1.55B 参数，支持 100+ 种语言，在大多数语言上的 Word Error Rate（WER，词错误率，识别文本和真实文本逐词比对后的错误比例，ASR 圈的核心精度指标，越低越好）都能做到很低。

但原始的 Whisper 推理非常慢。一段 30 秒的音频，用 large-v3 在 A100 上推理需要约 3-5 秒。这对离线转录还行，但对实时场景完全不够用。

**faster-whisper** 是目前生产环境的首选方案。它用 CTranslate2（一个针对 Transformer 模型的高性能 C++ 推理引擎，主打 CPU/GPU 上的低延迟和量化部署）引擎重新实现了 Whisper，相比原版：

- 推理速度快 4 倍
- 显存占用更低
- 支持 int8/float16 量化
- API 兼容原版 Whisper

用 faster-whisper 部署一个语音识别服务：

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# 转录一段音频
segments, info = model.transcribe("audio.wav", beam_size=5)
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

**生产部署的关键参数**：

- `compute_type`：float16 是速度和精度的最佳平衡点，int8 可以进一步省显存但可能有精度损失（int8/float16 等数值精度概念已在第 7 章量化里展开）
- `beam_size`（束搜索宽度，解码时同时保留多少个候选序列，越大越准但越慢）：beam_size=5 是默认值，设为 1 可以快 30% 但精度下降
- `vad_filter`：开启 VAD（Voice Activity Detection，语音活动检测，判断一段音频里是否有人声，用来切掉静音段）过滤，自动跳过静音段，对长音频效果显著

Whisper large-v3 turbo 是 2024 年底发布的精简版，参数量更小但速度快很多，适合对延迟敏感的场景。有团队报告在优化后实现了亚秒级延迟。

**中文场景的补充选项**：Whisper 在中文普通话上能用，但方言、粤语和带口音的语音 WER 偏高。国内生产环境推荐先评测以下两个开源方案：

- **FunASR**（阿里开源的中文 ASR 工具包，[github.com/modelscope/FunASR](https://github.com/modelscope/FunASR)）：中文普通话 WER 普遍低于 Whisper large-v3，推理速度也更快，是中文 ASR 的工业标准之一。
- **SenseVoice**（阿里 FunAudioLLM 团队开源的多功能语音理解模型，[github.com/FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)）：除识别外还支持情感识别、语种检测，适合需要这些副信号的对话场景。

如果你的用户群以中文为主，优先评测这两个再决定是否还需要 Whisper。

### TTS 模型：语音合成的部署

语音合成（Text-to-Speech，TTS，把文本转成自然语音的技术，和 ASR 是反向问题）这几年进步巨大。从早期的拼接合成到现在的神经网络 TTS，合成质量已经接近真人。

当前主流的开源 TTS 方案：

- **VITS / VITS2**（Variational Inference with adversarial learning for end-to-end Text-to-Speech，端到端 TTS 模型，把文本到声波的全流程放进一个网络里训）：推理速度快，适合实时场景
- **CosyVoice**（阿里 FunAudioLLM 团队的开源 TTS 模型，[github.com/FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)）：支持中英文，音色克隆（voice cloning，用一小段参考音频复刻出该说话人的音色）效果好
- **ChatTTS**（专为对话场景优化的开源 TTS，[github.com/2noise/ChatTTS](https://github.com/2noise/ChatTTS)）：支持韵律控制
- **Fish Speech**（开源社区出品的多语言 TTS，[github.com/fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)）：支持多语言和音色克隆

TTS 部署的核心指标是 **Real-Time Factor（RTF，实时率，合成 N 秒音频所需的计算秒数除以 N，RTF=1 表示刚好实时）**：合成 1 秒音频需要多少秒计算时间。RTF < 1 意味着能实时合成，RTF < 0.3 才算舒适。

以 CosyVoice 为例，最小可跑的服务化包装：

```python
# pip install cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio

# 加载模型（首次会自动下载权重）
tts = CosyVoice("iic/CosyVoice-300M-SFT")

def synthesize(text: str, speaker: str = "中文女", out_path: str = "out.wav") -> str:
    """文本 → 音频文件。speaker 是预置音色 ID。"""
    # generator 形式，支持流式输出，循环里就能一边合成一边播放
    for chunk in tts.inference_sft(text, speaker, stream=False):
        torchaudio.save(out_path, chunk["tts_speech"], 22050)
        return out_path

synthesize("你好，欢迎使用 CosyVoice 语音合成服务。")
```

生产部署里把这段逻辑包成 FastAPI 接口即可：接收文本，返回 WAV 字节或音频块流。注意 `stream=True` 模式下 generator 会持续 yield 出小块音频，可以直接通过 WebSocket 推给前端，做边合成边播。

### 实时语音对话的架构

实时语音对话是 2025 年最热门的应用场景之一。经典的架构是三段式 pipeline（流水线，把多个处理环节串成一条链路，每一步消费上一步的输出）：

```
用户语音 → STT（Speech-to-Text，语音转文字，和 ASR 是同义词的不同叫法）→ LLM（文本生成）→ TTS（语音合成）→ 播放给用户
```

每一段都有延迟，累加起来就是用户感知到的响应时间。人类对话的自然响应窗口是 300-500ms，超过 1 秒就会觉得"卡"。

一个典型的延迟预算（下表数值来自 2025-2026 年实时语音项目的工程实践综合，不是单一产品的实测值——具体数字会随 GPU 型号、模型规模和网络环境波动）：

| 环节 | 参考延迟（可达水平） | 说明 |
|------|---------|------|
| VAD + 音频采集 | ~50ms | 检测用户是否说完 |
| STT 转录 | ~150ms | Streaming STT，不等说完就开始转 |
| LLM TTFT | ~400ms | 第一个 token 的生成时间 |
| TTS 首个音频块 | ~150ms | 流式合成，不等全部文本就开始 |
| 网络开销 | ~50ms | 各环节间的传输 |
| **总计** | **~800ms** | 还是偏高，但可接受 |

**关键优化手段**：

1. **全链路流式处理**：STT 输出 partial transcript（部分转录，不等用户说完就先吐出已识别到的片段，类比 SSE 流式输出）就送给 LLM，LLM 输出几个 token 就送给 TTS。不要等任何一个环节完全结束。
2. **Streaming STT**（流式语音识别，边收音边识别）：用 WebSocket 持续发送音频片段，实时返回部分转录结果。可以比等用户说完快 200-400ms。
3. **投机执行**（speculative execution，借鉴 CPU 的概念，在结果尚未确定时提前开始下一步计算）：根据部分转录预测可能的回复，提前开始生成。用户说"帮我查一下明天北京的..."，不用等后面就可以开始准备天气查询。
4. **端到端语音模型**：跳过 STT→LLM→TTS 的三段式 pipeline，直接做 Speech-to-Speech（语音直接到语音，不经过中间文本表示，模型一次性输入音频输出音频）。OpenAI 的 GPT-4o（OpenAI 的原生多模态模型，文本、图片、音频共享同一个模型主干）语音模式、开源的 Ultravox（基于 Llama 的实时语音对话模型）和 Moshi（Kyutai 团队开源的端到端语音对话模型）都在走这条路，延迟可以压到 200-300ms。

端到端模型是未来趋势，但目前三段式 pipeline 在可控性和调试便利性上仍有优势——你可以单独升级任何一个环节。

## 15.3 多模态 Agent 的 Infra 挑战

当 Agent 需要同时处理图片、语音、视频和文本时，Infra 的复杂度会指数级上升。

### 多模态输入的预处理 pipeline

一个多模态 Agent 收到一个请求，可能包含：
- 一段文本指令
- 两张图片
- 一段 30 秒的语音

这些需要不同的预处理：

```python
# 伪代码：多模态预处理 pipeline
async def preprocess(request):
    tasks = []

    if request.images:
        # 图片：resize、归一化、转 tensor
        tasks.append(process_images(request.images, max_resolution=1344))

    if request.audio:
        # 音频：重采样到 16kHz、转文本（STT）
        tasks.append(transcribe_audio(request.audio))

    if request.video:
        # 视频：抽帧、每帧做图片预处理
        tasks.append(extract_and_process_frames(request.video, fps=1))

    # 并行执行所有预处理
    results = await asyncio.gather(*tasks)
    return merge_results(results, request.text)
```

关键设计原则：**各模态的预处理应该并行执行**。图片 resize 和音频 STT 之间没有依赖关系，串行处理是浪费。

### 不同模态的延迟差异

各模态的预处理延迟差异巨大：

| 模态 | 预处理延迟 | 生成的 token 数 |
|------|-----------|---------------|
| 文本 (500 字) | <10ms | ~500 tokens |
| 图片 (1344×1344) | 50-200ms | ~1000-9000 tokens |
| 音频 (30s) | 500ms-3s | ~100 tokens (转文本后) |
| 视频 (10s, 1fps) | 1-5s | ~10000+ tokens |

这种差异决定了你不能用统一的 timeout 和 rate limit 策略。一个包含视频的请求可能需要 10 秒预处理，而纯文本请求只需要几毫秒。

**实际做法**：按模态分配不同的请求队列和超时策略。轻量请求（纯文本）走快速通道，重量请求（视频）走独立队列，避免互相阻塞。

### 显存管理：图片 token 的显存开销

VLM 推理中最容易踩坑的就是显存管理。

一个请求带一张 1344×1344 的图片，可能产生 9000+ 个 visual tokens。在 7B 模型中，每个 token 在 KV Cache 中占用约 0.5MB（取决于模型结构和精度），9000 个 token 就是 4.5GB。单个请求就可能吃掉半张 A100 的显存。

vLLM 的 PagedAttention 在这种场景下尤为重要——它能按 page 粒度管理 KV Cache，避免预分配整块显存。但即使如此，你也需要：

1. **限制单个请求的最大 token 数**（包括 visual tokens）
2. **动态调整 batch size**：有图片请求时减少 batch 中的请求数
3. **提前计算 visual token 数量**：在请求进入推理引擎前就知道它会占多少显存，避免 OOM（Out Of Memory，显存/内存耗尽错误，发生时 GPU 直接拒绝分配，整个推理进程容易崩）

### 多模态 Embedding 和检索

RAG（Retrieval-Augmented Generation，检索增强生成，先从知识库检索相关片段再拼进 prompt 给 LLM，详见第 14 章）系统遇到多模态就更有意思了。传统的 text embedding（文本向量模型，把一句话编码成定长向量，用向量相似度做检索）只能处理文本，但现在你的知识库可能包含图片、图表、PDF 扫描件。

当前的多模态 embedding 方案：

- **CLIP 系列**（Contrastive Language-Image Pre-training，OpenAI 提出的图文对比预训练范式，是绝大多数多模态模型的视觉基础）：OpenAI 的 [CLIP](https://arxiv.org/abs/2103.00020)（开源权重 [huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)）通过对比学习把图片和文本训练到同一向量空间——用文本 embedding 可以直接检索图片，反之亦然，这是「图文同检」能成立的基础。Google 的 [SigLIP](https://arxiv.org/abs/2303.15343)（Sigmoid Loss for Language-Image Pre-training，把 CLIP 的 softmax 对比损失换成 sigmoid，训练更稳更省显存）是改进版，训练效率更高，越来越多场景在替换原版 CLIP。
- **ColPali / ColQwen**：专门为文档检索设计的多模态 embedding 模型。[ColPali](https://arxiv.org/abs/2407.01449)（[模型 ckpt](https://huggingface.co/vidore/colpali-v1.2)）用 PaliGemma（Google 开源的轻量 VLM，融合 SigLIP 视觉编码器和 Gemma 语言模型）VLM 直接对 PDF 页面图像做 late interaction（晚期交互，查询向量和文档每个 token 向量分别打分再聚合，比单向量检索更细粒度，源自 ColBERT）检索，能处理表格、公式、扫描件等 OCR（Optical Character Recognition，光学字符识别，把图片里的文字识别成可编辑文本，比如百度文字识别、Tesseract 这一类）难以可靠识别的内容；ColQwen 是基于 Qwen2-VL 的改进版本。
- **BGE-Visualized**：BAAI（Beijing Academy of Artificial Intelligence，北京智源人工智能研究院，BGE 系列向量模型的发布方）的多模态 embedding 模型，支持图文混合检索

多模态检索的 Infra 注意点：

1. **向量维度更高**：多模态 embedding 通常是 768-1024 维，比 text embedding 的 384 维大不少，存储和检索成本都更高
2. **索引构建更慢**：图片 embedding 需要 GPU 计算，建索引的速度比纯文本慢 10-100 倍
3. **查询也更贵**：如果查询本身包含图片，也需要 GPU 做 encoding

实际架构中，多模态 embedding 服务通常独立部署，和推理服务分开。这样可以独立伸缩——embedding 计算是批量型的，适合用大 batch 提高 GPU 利用率。

---

## 小结

多模态推理给 Infra 带来的挑战，本质上是**输入的不确定性大幅增加**。纯文本时代，一个请求的 token 数大概在几百到几千之间，相对可预测。多模态之后，一张高分辨率图片就可能带来上万个 token，一段视频更是难以预测。

这要求我们在调度、显存管理、限流和扩缩容策略上都做出相应调整。好消息是，vLLM 等框架已经在快速跟进，很多底层的复杂性正在被抽象掉。但理解这些机制，在出问题时才能快速定位和解决。

下一章，我们换一个完全不同的话题——聊聊作为 Agent 开发者，如何规划自己向 Infra 工程师转型的路径。

## 代码示例

| 示例 | 说明 | 硬件要求 |
|------|------|---------|
| [01_vlm_inference.py](../../examples/ch15-multimodal/01_vlm_inference.py) | VLM 图片推理示例 | GPU (any) |
| [02_whisper_server.py](../../examples/ch15-multimodal/02_whisper_server.py) | Whisper 语音识别服务 | GPU (any) |
| [03_multimodal_pipeline.py](../../examples/ch15-multimodal/03_multimodal_pipeline.py) | 多模态预处理 pipeline | GPU (any) |


---

> 本章来自《LLM Infra 从入门到实践》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-llm-infra](https://github.com/diguike/book-llm-infra)
