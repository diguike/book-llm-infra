# 第 16 章 延伸学习地图

读到这里你已经过了一遍 LLM Infra 的主干。但 Infra 是个深井，本书覆盖的是「让你能上手干活」的那一层，再往下还有大量内容没写——有的是篇幅原因，有的是话题本身还在快速变化、写进书就过期。

这一章不是顺序读的章节，是**一份地图 + 一份查询表**。需要哪一段拿出来翻就行。

## 16.1 怎么用这一章

本章五节按用途分：

- **16.2 各章对应的进阶资料**：每章给 2-4 个再往下读什么的入口（源码 / 论文 / 文档）。读完正文后想立刻深入某章时用这一节。
- **16.3 这本书没覆盖的高阶话题**：本章核心。十个方向，每个方向几句话点明它解决什么、入口在哪。读完想知道"我还差什么"时用这一节。
- **16.4 学习路径与节奏**：三档时间表（入门 / 系统 / 深度），不是规定动作，是参考节奏。
- **16.5 怎么读源码 / 怎么贡献**：vLLM / SGLang / llama.cpp 这种大项目的入门策略，外加渐进式 PR（Pull Request，向开源仓库提交代码变更的请求，类似 GitLab 的 MR）路径。
- **16.6 持续跟踪这个领域**：课程、博客、会议、个人作者——你需要订阅的几个信息源。

下面五节互相独立，你按需要跳着读。

## 16.2 各章对应的进阶资料

正文是入门通道，下面是每章再深入一层的几个入口。**优先级标注**：⭐ 必读、👍 推荐、📖 参考。

| 章 | 进阶资料 |
|---|---|
| **ch00 Python 环境** | ⭐ [`uv` 文档](https://docs.astral.sh/uv/)（Rust 写的新一代 Python 包管理器，下一步替代 pip/conda）；👍 [PEP 668](https://peps.python.org/pep-0668/)（Python 改进提案 668，理解 externally-managed env 这种保护系统 Python 的机制） |
| **ch01 全景** | ⭐ [Chip Huyen — Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html)；👍 OpenAI / Anthropic 的 [Cookbook](https://github.com/openai/openai-cookbook) |
| **ch02 Transformer** | ⭐ [Attention Is All You Need](https://arxiv.org/abs/1706.03762)；⭐ [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)；⭐ [Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)；👍 [nanoGPT 源码](https://github.com/karpathy/nanoGPT) |
| **ch03 GPU 基础** | ⭐ [Stanford CS149 Parallel Computing](https://cs149.stanford.edu/)；👍 [NVIDIA H100 白皮书](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)；👍 [GPU MODE Lectures](https://github.com/gpu-mode/lectures) |
| **ch04 推理基础** | ⭐ [Hugging Face Transformers 源码 `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)；👍 [Generation strategies 文档](https://huggingface.co/docs/transformers/generation_strategies) |
| **ch05 vLLM** | ⭐ [PagedAttention 论文 (SOSP 2023)](https://arxiv.org/abs/2309.06180)；⭐ [vLLM 源码 `core/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py)；👍 [vLLM Office Hours YouTube](https://www.youtube.com/@vllm-project) |
| **ch06 引擎对比** | ⭐ [SGLang 论文](https://arxiv.org/abs/2312.07104)；👍 [TensorRT-LLM 性能 benchmark](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-overview.md) |
| **ch07 量化** | ⭐ [GPTQ](https://arxiv.org/abs/2210.17323) / [AWQ](https://arxiv.org/abs/2306.00978) / [SmoothQuant](https://arxiv.org/abs/2211.10438)；👍 [llm-compressor](https://github.com/vllm-project/llm-compressor)；📖 [bitsandbytes 源码](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| **ch08 推理加速** | ⭐ [FlashAttention v1](https://arxiv.org/abs/2205.14135) / [v2](https://arxiv.org/abs/2307.08691) / [v3](https://arxiv.org/abs/2407.08608)；⭐ [FlashInfer](https://github.com/flashinfer-ai/flashinfer)；👍 [Tri Dao 博客](https://tridao.me/) |
| **ch09 微调** | ⭐ [LoRA](https://arxiv.org/abs/2106.09685) / [QLoRA](https://arxiv.org/abs/2305.14314) 论文；⭐ [TRL 文档](https://huggingface.co/docs/trl)；👍 [Unsloth](https://github.com/unslothai/unsloth)（QLoRA 加速）；👍 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| **ch10 对齐** | ⭐ [DPO 论文](https://arxiv.org/abs/2305.18290)；👍 [Lilian Weng — RLHF Survey](https://lilianweng.github.io/posts/2024-08-31-rlhf-pitfalls/)；📖 [InstructGPT](https://arxiv.org/abs/2203.02155) |
| **ch11 分布式训练** | ⭐ [ZeRO 论文](https://arxiv.org/abs/1910.02054)；⭐ [PyTorch FSDP 文档](https://pytorch.org/docs/stable/fsdp.html)；👍 [Megatron-LM 源码](https://github.com/NVIDIA/Megatron-LM)；📖 [DeepSpeed 文档](https://deepspeed.readthedocs.io/) |
| **ch12 生产部署** | ⭐ [vLLM Production Stack](https://github.com/vllm-project/production-stack)；👍 [KServe](https://kserve.github.io/)；👍 [Ray Serve LLM 教程](https://docs.ray.io/en/latest/serve/llm/serving-llms.html) |
| **ch13 可观测 / 成本** | ⭐ [Langfuse](https://github.com/langfuse/langfuse)（自部署）；👍 [Helicone](https://github.com/Helicone/helicone)；👍 [OTel GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) |
| **ch14 RAG** | ⭐ [LlamaIndex 文档](https://docs.llamaindex.ai/)；⭐ [RAG Survey 2024](https://arxiv.org/abs/2312.10997)；👍 [GraphRAG (Microsoft)](https://github.com/microsoft/graphrag) |
| **ch15 多模态** | ⭐ [LLaVA](https://github.com/haotian-liu/LLaVA)；⭐ [Qwen2-VL 技术报告](https://arxiv.org/abs/2409.12191)；👍 [Whisper](https://github.com/openai/whisper)；👍 [FunASR](https://github.com/modelscope/FunASR)（中文 ASR，Automatic Speech Recognition，自动语音识别） |

## 16.3 这本书没覆盖的高阶话题

下面十个方向，每个都能再写一本书。这里不展开，只给「它解决什么 + 入口在哪」，便于你自己沿着线索往下扒。

### 推理优化进阶：Speculative Decoding 家族

ch08 讲了 Speculative Decoding 的基本思路（小模型出草稿 + 大模型并行验证），但 2024 年起这个方向爆发了一批工程化变体：

- **EAGLE / EAGLE-2 / EAGLE-3**：用大模型自身的 hidden states 出草稿，比独立 draft 模型显存省、命中率高。论文：[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840)。
- **Medusa**：在大模型上加多个并行预测头，一次出多个 token 候选。[GitHub](https://github.com/FasterDecoding/Medusa) · [论文](https://arxiv.org/abs/2401.10774)。
- **Lookahead Decoding**：完全不需要 draft 模型，用 Jacobi 迭代生成候选。[论文](https://arxiv.org/abs/2402.02057)。

vLLM v0.6+ 已经原生集成了 EAGLE 和 Medusa，看 [vLLM speculative decoding 文档](https://docs.vllm.ai/en/latest/features/spec_decode.html) 是上手最快的路径。

### 长上下文工程：RoPE 缩放、StreamingLLM、稀疏 Attention

模型训练时只见过 4K-32K 的上下文，怎么让它处理 100K+？这是一个独立的工程方向：

- **RoPE 缩放系列**：[Position Interpolation](https://arxiv.org/abs/2306.15595)（NTK aware）、[YaRN](https://arxiv.org/abs/2309.00071)、[LongRoPE](https://arxiv.org/abs/2402.13753)。原理是修改位置编码让模型「假装」从未见过的长度也是熟悉的。
- **StreamingLLM**：保留最早几个 token（attention sink）+ 滑动窗口，实现无限上下文。[论文](https://arxiv.org/abs/2309.17453) · [GitHub](https://github.com/mit-han-lab/streaming-llm)。
- **稀疏 Attention**：[H2O](https://arxiv.org/abs/2306.14048)、[SnapKV](https://arxiv.org/abs/2404.14469)、[MInference](https://arxiv.org/abs/2407.02490) 等，通过动态裁剪 KV Cache 把长上下文显存压下来。
- **环形注意力 / Ring Attention**：把超长序列切分到多卡上做 attention，单卡看不到全序列但通过通信拼起来。[论文](https://arxiv.org/abs/2310.01889)。

### LoRA Serving 工程化

ch09 讲了怎么训 LoRA，但**生产环境同时 serve 几十个 LoRA adapter** 是另一个问题：每个 adapter 几 MB，但每次切换都得加载，怎么做到秒级切换、批量请求里混用不同 adapter？

- **vLLM Multi-LoRA**：原生支持，启动时 `--enable-lora --max-loras N`。[文档](https://docs.vllm.ai/en/latest/features/lora.html)。
- **S-LoRA**：把多个 LoRA adapter 的计算融合，吞吐量提升 4x。[论文](https://arxiv.org/abs/2311.03285) · [GitHub](https://github.com/S-LoRA/S-LoRA)。
- **Punica**：专门为 multi-tenant LoRA 设计的 serving 系统。[论文](https://arxiv.org/abs/2310.18547) · [GitHub](https://github.com/punica-ai/punica)。

如果你的业务场景是 SaaS（每个客户一个微调 adapter），这是绕不开的方向。

### 推理服务网格

ch12 讲的是单个 vLLM 实例的部署。当你有几十上百个推理实例、需要按租户路由、按模型路由、按 prefix cache 路由时，需要**LLM 专用的 router**：

- **vLLM Production Stack**：vLLM 官方的多实例编排栈，含 router、cache、observability。[GitHub](https://github.com/vllm-project/production-stack)。
- **AIBrix**（字节）：K8s native 的 LLM serving 平台。[GitHub](https://github.com/vllm-project/aibrix)。
- **SGLang Router**：SGLang 自带的 prefix-aware 路由器，按 prompt 前缀分发到缓存命中率高的实例。[文档](https://docs.sglang.ai/router/router.html)。
- **LiteLLM Proxy**：跨多家云 API + 自建模型的统一网关。[GitHub](https://github.com/BerriAI/litellm)。

### 高级 RAG：GraphRAG、Agentic、ColPali

ch14 覆盖了主流 RAG 架构，但下面几个方向是 2024-2026 年的活跃前沿：

- **GraphRAG**（Microsoft）：先用 LLM 从文档抽取实体和关系建图，检索时同时查图谱和向量。对"全局问题"（如"这本书的主旨是什么"）显著优于纯向量检索。[GitHub](https://github.com/microsoft/graphrag) · [论文](https://arxiv.org/abs/2404.16130)。
- **Agentic RAG**：把检索从「固定一步」变成 Agent 工具，让模型自己决定查什么、查几次、是否回查。[LangGraph 教程](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)。
- **ColPali**：用视觉模型直接 embed PDF 页面图像，跳过文本提取这一步。对扫描件、复杂排版 PDF 是质的飞跃。[论文](https://arxiv.org/abs/2407.01449) · [GitHub](https://github.com/illuin-tech/colpali)。
- **Matryoshka Embeddings**：训练时让 embedding 的不同前缀长度都各自可用，检索时先用短前缀粗筛再用全量精排，速度提升数倍。[论文](https://arxiv.org/abs/2205.13147)。

### 扩散模型 Infra（图像 / 视频生成）

本书完全没碰扩散模型，但 LLM Infra 工程师转去做 Stable Diffusion / Sora-like 系统的需求量很大。共性是 GPU 推理优化，差异在：

- **架构差异**：UNet → DiT（Diffusion Transformer）已经成新主流，[Sora](https://openai.com/research/video-generation-models-as-world-simulators)、[Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3) 都用 DiT。
- **采样器优化**：DDIM / DPM-Solver / Flow Matching——决定生成需要多少步。[Flow Matching 论文](https://arxiv.org/abs/2210.02747)。
- **VAE 加速**：图像扩散里 VAE 编解码是瓶颈，[TAESD](https://github.com/madebyollin/taesd) 等轻量替代方案值得关注。
- **生产部署**：[ComfyUI](https://github.com/comfyanonymous/ComfyUI) 是事实标准，[Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) 是性能优化分支。

### 分布式预训练深度

ch11 讲了 DDP / ZeRO / FSDP 这些**用得到**的并行策略，但真正训一个百亿参数以上的模型需要的是 **3D Parallelism**（三维并行，把数据并行 DP、张量并行 TP、流水线并行 PP 三种切分方式叠加使用） 全套：

- **Megatron-LM**：NVIDIA 出品，3D 并行的事实标准实现。[GitHub](https://github.com/NVIDIA/Megatron-LM) · [论文](https://arxiv.org/abs/1909.08053)。
- **Megatron-Core**：把 Megatron-LM 的并行原语剥出来做成库，可以嵌入其他训练框架。[GitHub](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core)。
- **Colossal-AI**：另一套国产开源训练系统，支持 ZeRO+TP（Tensor Parallelism，张量并行）+PP（Pipeline Parallelism，流水线并行）。[GitHub](https://github.com/hpcaltech/ColossalAI)。
- **NeMo Framework**：NVIDIA 的端到端训练 + 对齐 + 部署平台。[文档](https://docs.nvidia.com/nemo-framework/)。

如果你要参与从零训百亿参数的项目，这是必经之路。

### 联邦 / 隐私推理

涉及金融、医疗、政府场景时，模型权重和用户输入都不能离开机房。这个方向 2025-2026 年开始有工业级方案：

- **TEE（Trusted Execution Environment）**：NVIDIA H100/H200 支持 [Confidential Computing](https://www.nvidia.com/en-us/data-center/solutions/confidential-computing/)，能在加密内存里跑推理。Intel TDX 也是同方向。
- **同态加密推理**：理论上完美，工程上还很慢。代表项目 [Concrete-ML](https://github.com/zama-ai/concrete-ml)。
- **安全多方计算（MPC，Multi-Party Computation）**：把模型和数据切分到多方各算一部分，任何单方都看不到完整数据。[CrypTen (Meta)](https://github.com/facebookresearch/CrypTen)。
- **本地推理 + 端云协同**：把敏感请求路由到本地小模型，其他走云端大模型。这是更工程化的折衷方案。

### Compiler Stack：torch.compile / TVM / IREE / MLIR

为什么要关心编译器？因为推理引擎的下一波优化都在这里——把 PyTorch 算子图编译成针对特定硬件优化的代码：

- **torch.compile**：PyTorch 2.0+ 内置，前端 [TorchDynamo](https://pytorch.org/docs/stable/torch.compiler.html) 抓图，后端 [Inductor](https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html) 生成 Triton kernel。**最值得先学**。
- **Triton**：OpenAI 出品的 GPU kernel DSL（Domain-Specific Language，领域专用语言，专门为某类问题设计的小型编程语言），Python 语法，比 CUDA C 友好得多。FlashAttention 的可读版本就是 Triton 写的。[文档](https://triton-lang.org/)。
- **TVM**：Apache 项目，编译到多种硬件后端（CPU/GPU/ARM/NPU，Neural Processing Unit，神经网络处理器，专为 AI 推理设计的芯片）。[官网](https://tvm.apache.org/)。
- **IREE / MLIR**：Google + LLVM（一套通用编译器基础设施）阵营的 ML（Machine Learning，机器学习）编译器栈，往下打通到硬件层。[IREE](https://iree.dev/)。

### 数据飞轮：用户反馈如何回流到训练

这个话题模糊但极重要——决定了你的模型能不能持续进化：

- **隐式反馈收集**：用户点了哪个候选、重复了哪个问题、停留多久——这些信号比"赞 / 踩"密度高 100 倍。需要在推理服务层埋点。
- **DPO/SimPO 数据 pipeline**：怎么把"用户最终采用的回答"自动转成 chosen-rejected pair，进入下一轮微调。
- **在线学习架构**：训练侧和推理侧的版本同步、A/B 实验框架、灰度回滚。[Anthropic Constitutional AI](https://arxiv.org/abs/2212.08073) 的 RLAIF（Reinforcement Learning from AI Feedback，从 AI 反馈中强化学习，相对 RLHF 用人类反馈，这里用另一个 AI 模型当评审）思路值得参考。
- **数据治理**：日志怎么脱敏、用户授权怎么管理、训练集怎么去重——离合规越近，工程量越大。

### 专用硬件：Groq、Cerebras、Tenstorrent

NVIDIA 不是唯一选项。下面几家在 2024-2026 年已经能跑生产 workload：

- **[Groq LPU](https://groq.com/)**（Language Processing Unit，专为 LLM 推理设计的芯片）：极低 TTFT（毫秒级），适合实时对话。模型规模有限制。
- **[Cerebras WSE](https://www.cerebras.net/)**：晶圆级集成（一片硅 = 一个芯片），训练超大模型用。
- **[Tenstorrent](https://tenstorrent.com/)**：RISC-V + Tensix 核，开源软件栈，价格亲民。
- **[AMD MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)**：vLLM、SGLang 都已经支持 ROCm（AMD 对标 CUDA 的 GPU 计算平台）后端，部分场景性价比胜 H100。

如果你的公司在做"非 NVIDIA 备选方案"，这条线值得跟。

## 16.4 学习路径与节奏

下面三档不是 prescriptive，是**参考节奏**。每个人投入时间不同，挑你能稳住的那一档。

### 入门档（一周）

目标：能跑通一个推理服务、看懂监控面板、和团队聊技术名词不卡。

```
Day 1-2: 读完 ch00-ch01，配好双环境
Day 3:   ch04（推理基础）+ ch05（vLLM）—— 跑通本地一个 7B 模型的服务
Day 4:   ch06 引擎对比 + ch07 量化 —— 自己换一个量化版跑一遍
Day 5:   ch12 部署 + ch13 可观测 —— 把服务装到一个 K8s 集群里加监控
Day 6-7: ch14 RAG —— 跟一个 LlamaIndex 教程做完一个 demo
```

跳过 ch02-03（除非你想读 Transformer）、ch08-11（除非你要做训练 / 优化）。

### 系统档（一个月）

目标：理解每个章节的 why，能独立做技术选型，能解决日常 on-call（值班响应线上故障）问题。

```
Week 1: ch00-04，跑通所有代码示例，重点理解 Attention 和 KV Cache
Week 2: ch05-08，深入推理引擎，自己跑一遍 GPTQ/AWQ 量化
Week 3: ch09-11，跑一遍 QLoRA 微调 + DPO 对齐
Week 4: ch12-15，部署一套带可观测的服务 + 做一个 RAG 系统
```

整个过程中，不要追求"读完所有论文"——遇到不懂的概念查一下就跳过，绝大多数概念在工程上够用就行。

### 深度档（半年）

目标：能给某个开源项目持续贡献、能优化 kernel、能独立 own 一个 Infra 子方向。

```
月 1-2:  按系统档过一遍正文 + 16.2 各章进阶资料
月 3-4:  挑 1-2 个开源项目深入读源码（vLLM / SGLang / llama.cpp 三选一或二）
         开始提 Level 1-2 的 PR（见 16.5）
月 5:    挑 16.3 里的 1-2 个高阶话题深入（推荐 Compiler Stack 或 Speculative Decoding）
月 6:    自己实现一个端到端的优化项目（比如：把某模型的 TTFT 优化 30%）
```

**PyTorch 学习路径**（嵌入深度档第 1-2 个月）：

1. 第一周：用 transformers 加载 Qwen2.5-0.5B，理解 `model.generate()` 参数，用 `torch.profiler` 看推理时 GPU 在干什么。
2. 第二周：读 transformers 的 `modeling_qwen2.py`，理解 Attention、MLP、RMSNorm，加 `print(tensor.shape)` 看维度流动。
3. 第三到四周：Fork vLLM 本地跑起来，改一个简单配置项（比如 batch scheduler 参数），读 KV Cache 管理代码。
4. 之后：在实际项目中持续加深，需要什么学什么。

**核心原则**：Infra 工程师需要的 PyTorch 知识和 ML 研究员（Machine Learning Researcher，专注论文和新模型的研究岗位，对应 Applied Scientist 偏应用一点）是不一样的。你不需要会写新模型架构，但需要理解现有模型的计算图、知道哪里是性能瓶颈、能读懂和修改模型代码。

**CUDA 学习节奏**（嵌入深度档第 3-5 个月）：

| 阶段 | 时长 | 目标 |
|---|---|---|
| 能读懂 | 1-2 月 | 学习 thread/block/grid/shared memory/warp 基本概念；读简单 kernel（矩阵乘法、softmax、RMSNorm）；目标：看到一个 kernel 能说出它在干什么 |
| 能改 | 3-6 月 | 拿现有 kernel 做优化（增加 shared memory 利用）；学 Triton（Python 语法，比 CUDA C 友好）；读 FlashAttention 的 Triton 实现 |
| 能写 | 6-12 月+ | 从零实现自定义 kernel；深入 GPU 架构（memory hierarchy、bank conflict、occupancy）；用 NSight Compute（NVIDIA 官方的 kernel 级 profiling 工具）做 profiling，理解 roofline model（屋顶线模型，判断 kernel 是受算力还是受内存带宽限制的分析方法） |

大多数 Infra 工程师的日常工作不需要从头写 CUDA kernel，但需要能**读懂**别人写的、必要时**能改**。不要急于求成——很多资深 Infra 工程师也是在工作中逐步积累 CUDA 经验的。

## 16.5 怎么读源码 / 怎么贡献

在 LLM Infra 领域，**开源贡献（OSS contribution，Open Source Software contribution，给开源项目提交代码、文档或测试）的学习价值远超教程和课程**。原因有三：

1. **面对真实工程问题**。教程和课程给你的是简化后的问题，开源项目里的 issue 才是真实世界的样子——边界情况、性能退化、兼容性问题。
2. **代码 review 是最高效的学习**。当 vLLM 的核心维护者 review 你的 PR 时，你获得的 feedback 质量远超任何课程。
3. **可验证的能力证明**。一个被 merge 的 vLLM PR 比简历上写"精通 CUDA"有说服力一百倍。

### 从哪些项目开始

**vLLM**（[GitHub](https://github.com/vllm-project/vllm)，截至 2026 年初 40k+ stars）
- 最成熟的 LLM 推理引擎
- Python 为主，核心调度逻辑可读性不错
- Issue 数量多，`good first issue` 标签的任务适合新手
- 社区活跃，PR review 速度快

**SGLang**（[GitHub](https://github.com/sgl-project/sglang)，10k+ stars）
- 性能导向的推理框架，部分 benchmark 超过 vLLM
- 团队更小，贡献者更容易被注意到
- 截至 2026 年初，对长期活跃贡献者提供 AI 编程工具赞助，具体见 Discord 公告
- 比 vLLM 更激进地采用新技术

**llama.cpp**（[GitHub](https://github.com/ggerganov/llama.cpp)，70k+ stars）
- 纯 C/C++ 实现，量化推理的事实标准
- 代码质量高，是学习底层实现的好教材
- 对 C++ 功底有一定要求
- 支持的硬件平台最多：CPU、CUDA、Metal、Vulkan

### 渐进式贡献路径

| Level | 阶段目标 | 典型工作 |
|---|---|---|
| **1. 文档和测试**（第 1-2 个 PR）| 熟悉 PR 流程、CI（Continuous Integration，持续集成，每次推代码就自动跑测试和检查）系统、代码规范 | 修文档错误 / 过时信息、补缺失的测试用例、改善错误信息可读性 |
| **2. Bug Fix**（第 3-5 个 PR）| 开始理解项目核心架构 | 从 issue 列表找已复现的 bug、边界情况处理、小的性能优化 |
| **3. Feature**（第 5-10 个 PR）| 对项目有较深理解 | 新增模型支持、实现社区讨论过的功能、参与 RFC（Request for Comments，征求意见稿，开源社区里大改动方案的公开讨论文档）讨论 |
| **4. Core**（持续）| 成为活跃贡献者 / 维护者 | 核心模块重构 / 优化、性能关键路径改进、review 别人的 PR、参与方向讨论 |

### 怎么读懂大型 C++/CUDA 项目

面对 llama.cpp 或 vLLM 的 CUDA 代码，很多人的第一反应是「完全看不懂」。几个实用技巧：

1. **从入口点开始**。不要试图从头到尾读代码，先找到 `main()` 或 API 入口，然后沿调用链往下追。
2. **用 Debug 模式跑**。编译 debug 版本，用 GDB（GNU Debugger，Linux 下的 C/C++ 调试器，类似 JS 里的 `node --inspect`）或 Python debugger 加断点，观察实际的执行路径和数据。
3. **画调用图**。用纸笔或工具画出核心函数的调用关系，几个核心模块搞清楚后，其他的就好理解了。
4. **读 Git 历史**。`git log --oneline --follow <file>` 看某个文件的变更历史，从早期的简单版本开始读，比读最新的复杂版本容易得多。
5. **善用 AI 工具**。把一段看不懂的代码丢给 Claude / Cursor，让它逐行解释。这不丢人，这是效率。

## 16.6 持续跟踪这个领域

这个领域每周都有新东西。不需要全部追，但下面几个信息源订阅起来，能让你在被新名词砸到时不慌。

### 课程

**必看**：

- **Andrej Karpathy — Neural Networks: Zero to Hero**（[karpathy.ai/zero-to-hero](https://karpathy.ai/zero-to-hero.html) · [YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)）：从反向传播讲到 GPT，理解 Transformer 最好的免费课程。前几集 + 最后的 GPT 实现就够了。
- **Stanford CS149 — Parallel Computing**（[cs149.stanford.edu](https://cs149.stanford.edu/)）：理解 GPU 并行计算的原理，对后续学 CUDA 帮助很大。课程主页公开可看。
- **GPU MODE Lectures**（[github.com/gpu-mode/lectures](https://github.com/gpu-mode/lectures)）：社区驱动课程，CUDA 基础到 Triton 优化，每讲都有代码。YouTube 和 Discord 都活跃。

**推荐**：

- **Stanford CS336 — Language Modeling from Scratch**（[stanford-cs336.github.io](https://stanford-cs336.github.io/spring2025/)）：2025 年新课，从训练到部署完整覆盖。
- **Hugging Face NLP Course**（[huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course)）：免费、实战导向，从 tokenizer 到训练都覆盖。

### 博客和社区

**必关注**：

- **vLLM Blog**（[blog.vllm.ai](https://blog.vllm.ai/)）：官方技术博客，新功能和设计决策的第一手资料
- **Hugging Face Blog**（[huggingface.co/blog](https://huggingface.co/blog)）：模型和推理相关的高质量技术文章
- **GPU MODE Discord**（[discord.gg/gpumode](https://discord.gg/gpumode)）：最活跃的 GPU 编程社区
- **r/LocalLLaMA**（[reddit.com/r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)）：本地部署 LLM 的社区，实战经验密度极高

**作者 / 团队**：

- [Tri Dao](https://tridao.me/)（FlashAttention 作者）
- [Woosuk Kwon](https://woosukkwon.com/)（vLLM 作者）
- [Lilian Weng — Lil'Log](https://lilianweng.github.io/)（前 OpenAI，系统性梳理 LLM 关键技术）
- [Sebastian Raschka — Ahead of AI](https://magazine.sebastianraschka.com/)（深度学习教学，节奏稳）
- [Simon Willison](https://simonwillison.net/)（应用层视角，新模型测评勤快）

### 论文与会议

不需要把"读论文"做成日常习惯，但下面几篇是绕不开的：

- **Transformer**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)（2017）
- **FlashAttention**：[v1](https://arxiv.org/abs/2205.14135) / [v2](https://arxiv.org/abs/2307.08691) / [v3](https://arxiv.org/abs/2407.08608)
- **PagedAttention**：[vLLM 论文](https://arxiv.org/abs/2309.06180)
- **量化方法**：[GPTQ](https://arxiv.org/abs/2210.17323) / [AWQ](https://arxiv.org/abs/2306.00978) / [SmoothQuant](https://arxiv.org/abs/2211.10438)

**会议**：

- **[MLSys](https://mlsys.org/)**（Conference on Machine Learning and Systems）：ML 系统的核心会议，每年的论文和视频都公开
- **[OSDI](https://www.usenix.org/conference/osdi24)**（Operating Systems Design and Implementation） / **[SOSP](https://sosp.org/)**（Symposium on Operating Systems Principles）：系统领域顶会，LLM Infra 的重要论文常在这里
- **NeurIPS / ICML / ICLR**：ML 三大会（神经信息处理系统大会 / 国际机器学习大会 / 国际表征学习大会），模型架构和训练方法的来源

不需要亲自参会，论文和视频都是公开的，跟你订 newsletter 一个节奏看就行。

---

*全书完。*


---

> 本章来自《LLM Infra 从入门到实践》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-llm-infra](https://github.com/diguike/book-llm-infra)
