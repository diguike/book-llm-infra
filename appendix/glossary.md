# 附录 C — 术语表

本术语表按主题分组，每个条目给出中文名、英文（含缩写）、一句话定义和主要出现章节，方便读者随时回查。同一主题内按英文字母顺序排列。

## 模型与架构

| 术语 | 英文 | 一句话定义 | 主要章节 |
|------|------|------|------|
| 激活参数 | Active Parameters | MoE 模型每个 token 实际计算用到的参数量，决定推理速度；和总参数对应 | ch02 |
| 注意力机制 | Attention | Transformer 核心计算，让每个 token 根据 Q/K/V 关注序列里的其他 token | ch02 |
| DeepSeek | DeepSeek | 国内 AI 公司开源的模型系列，技术路线偏激进，代表作 DeepSeek-V3 用 MoE 架构（671B 总参 / 37B 激活） | ch01、ch07 |
| 向量化 / 嵌入 | Embedding | 把 token、图片或音频映射成固定维度的浮点向量，语义相近的向量在空间中距离近 | ch02、ch14 |
| 前馈网络 | FFN (Feed-Forward Network) | Transformer Block 内的两层 MLP，负责对每个 token 做非线性特征变换 | ch02 |
| 分组查询注意力 | GQA (Grouped Query Attention) | 多个 Q 头共享同一组 K/V 头，显著减小 KV Cache 体积，Llama 2 70B / Llama 3 默认使用 | ch02、ch04 |
| Hugging Face | Hugging Face | LLM 生态最大的模型托管平台 + Transformers 库 + 配套训练/推理工具家族（PEFT、TRL、Accelerate 等） | 全书 |
| 键值缓存 | KV Cache | 缓存历史 token 的 Key/Value，Decode 阶段只算新 token 的 Q 即可，是 LLM 推理显存大头 | ch01、ch02、ch05、ch08 |
| Llama | Llama | Meta 开源的大语言模型系列，从 Llama 1 (2023) 到 Llama 4 (2025)，架构成为现代开源 LLM 事实标准 | ch01、ch02、全书 |
| 层归一化 | LayerNorm / RMSNorm | 对每层输出做归一化稳定训练；RMSNorm 是 Llama 系列用的简化版，只做 RMS 缩放 | ch02 |
| 大语言模型 | LLM (Large Language Model) | 基于 Transformer 的大规模预训练语言模型 | 全书 |
| Logits | Logits | 模型最后一层输出的未经 softmax 的原始分数向量，长度等于词表大小 | ch04 |
| Mistral | Mistral | 法国 Mistral AI 开源的模型系列，包含 Mistral 7B / Mixtral 8x7B (MoE) / Codestral 等，架构与 Llama 对齐 | ch01 |
| 混合专家 | MoE (Mixture of Experts) | 每个 token 只激活 FFN 内的部分专家子网络，参数量大但计算量小，DeepSeek-V3 / Mixtral 采用 | ch02、ch04 |
| 参数量 | Parameter Count (B / M) | 模型权重总数；B = Billion 十亿、M = Million 百万。1B 参数 ≈ 2 GB 显存 (FP16)，决定显存占用、推理速度和模型能力 | ch01、ch02、全书 |
| Qwen | Qwen | 阿里巴巴开源的通义千问模型系列，国内开源代表，架构与 Llama 对齐，多数版本 Apache 2.0 协议 | ch01、ch05 |
| 旋转位置编码 | RoPE (Rotary Position Embedding) | 通过对 Q/K 向量做角度旋转编码位置，相对加法编码更利于上下文外推 | ch02 |
| Softmax | Softmax | 把任意实数向量转成总和为 1 的概率分布 | ch02、ch04 |
| Token | Token | 模型处理的最小文本单位，通常是子词，通过 Tokenizer 编码得到 | ch01、ch02 |
| 分词器 | Tokenizer | 把文本切成 token 序列并映射成 ID 的工具，常见有 BPE / SentencePiece / tiktoken | ch02、ch04 |
| 总参数 | Total Parameters | 模型权重的总数，对 MoE 模型决定显存占用（所有专家都得装入显存），与「激活参数」对应 | ch02 |
| Transformer | Transformer | Attention is All You Need 提出的 Encoder-Decoder 架构，LLM 主流采用其 Decoder-only 变体 | ch02 |

## 推理与服务

| 术语 | 英文 | 一句话定义 | 主要章节 |
|------|------|------|------|
| 连续批处理 | Continuous Batching | 同一 batch 内不等所有请求完成，有 slot 就插入新请求，vLLM 调度核心 | ch01、ch05 |
| 切块预填 | Chunked Prefill | 把长 prompt 的 prefill 切成多个 chunk，和 decode 交织执行，避免阻塞 | ch05 |
| 解码 | Decode | 推理第二阶段，每步只生成 1 个 token，反复跑直到结束，memory-bound | ch01、ch03、ch04 |
| FlashAttention | FlashAttention | 利用 Shared Memory 分块计算的 Attention kernel，省显存又加速，FlashAttention 2/3 持续优化 | ch08 |
| FlashInfer | FlashInfer | 专为 LLM decode 优化的 attention 库，vLLM v0.6+ 默认集成 | ch08 |
| PagedAttention | PagedAttention | vLLM 提出的 KV Cache 内存管理方案，把显存切成固定大小的 block 用 page table 索引 | ch05 |
| 预填充 | Prefill | 推理第一阶段，一次性并行处理整个输入 prompt，compute-bound | ch01、ch03、ch04 |
| 前缀缓存 | Prefix Caching | 复用相同前缀（如系统 prompt）的 KV Cache，跨请求降低 TTFT | ch05 |
| 服务端推送事件 | SSE (Server-Sent Events) | HTTP 长连接单向推流协议，LLM 流式输出的事实标准 | ch04 |
| 投机解码 | Speculative Decoding | 用一个小 draft 模型先预测 N 个 token，再用大模型一次验证，吞吐提升 2-3 倍 | ch08 |
| 每输出 token 时间 | TPOT (Time Per Output Token) | 稳态 decode 阶段相邻两个 token 的间隔，约等于 1000/TPS（毫秒），决定流式输出流畅度 | ch08、ch13 |
| 吞吐量 | TPS (Tokens Per Second) | 单位时间生成的 token 总数，并发场景下的整体吞吐指标 | ch08、ch13 |
| 首 Token 延迟 | TTFT (Time to First Token) | 从请求到首个 token 返回的时间，等于 prefill 耗时加调度排队 | ch01、ch05、ch08、ch13 |

## 量化

| 术语 | 英文 | 一句话定义 | 主要章节 |
|------|------|------|------|
| AWQ | AWQ (Activation-aware Weight Quantization) | 基于激活值幅度筛选保护重要权重的 4-bit 后训练量化方法 | ch07 |
| BF16 | BF16 (Brain Float 16) | 16 位浮点格式，指数位和 FP32 相同，数值范围广，现代 LLM 训练默认 | ch03、ch07 |
| bitsandbytes | bitsandbytes | Hugging Face 集成的量化库，提供 `load_in_4bit` / `load_in_8bit` 快速推理 | ch07 |
| FP8 | FP8 | 8 位浮点，Hopper 架构（H100/H200）原生支持，精度接近 BF16，吞吐再翻倍 | ch07 |
| FP16 | FP16 (Half Precision) | 16 位浮点格式，精度高但数值范围窄，A100 之前是默认推理精度 | ch03、ch07 |
| GGUF | GGUF | llama.cpp 定义的量化模型权重格式，单文件包含权重 + tokenizer + 元数据 | ch04、ch07 |
| GPTQ | GPTQ | 基于二阶信息（Hessian 矩阵近似）的逐层后训练量化方法，4-bit 主流方案之一 | ch07 |
| INT4 / INT8 | INT4 / INT8 | 4 位 / 8 位整数量化，配合 Tensor Core 整数路径可大幅提升吞吐 | ch03、ch07 |
| 量化 | Quantization | 把权重从高精度（FP16/BF16）压缩到低精度（INT4/INT8/FP8）以节省显存和带宽 | ch07 |
| SmoothQuant | SmoothQuant | 通过对激活值做缩放迁移减少 outlier 影响的 INT8 量化方法，已集成进 llm-compressor | ch07 |

## 训练与对齐

| 术语 | 英文 | 一句话定义 | 主要章节 |
|------|------|------|------|
| 交叉熵损失 | Cross-Entropy Loss | SFT 训练目标，最小化 `-log(实际 token 的预测概率)`，仅在 response 部分计入 | ch09 |
| 直接偏好优化 | DPO (Direct Preference Optimization) | 用排序对数据直接优化策略模型，绕开 reward model，比 PPO 简单稳定 | ch10 |
| 梯度累积 | Gradient Accumulation | 累积 N 步梯度再更新一次参数，等价于 N 倍 batch size 但不增加显存 | ch09、ch11 |
| 梯度检查点 | Gradient Checkpointing | 反向传播时不保存全部激活值，用时间换空间，显存节省 30-60% | ch09 |
| KL 散度 | KL Divergence | 衡量两个概率分布"有多不同"的指标，RLHF/DPO 中用于约束新模型不偏离 SFT 模型太远 | ch10 |
| 低秩适配 | LoRA (Low-Rank Adaptation) | 冻结原模型权重，旁挂低秩矩阵 A·B 参与训练，显著降低微调显存和存储 | ch09 |
| 近端策略优化 | PPO (Proximal Policy Optimization) | RLHF 阶段经典算法，同时维护 actor / critic / reference / reward 四个模型 | ch10 |
| 量化低秩适配 | QLoRA | 把原模型量化到 4-bit 再做 LoRA 微调，单卡 24GB 显存可微调 7B 模型 | ch09 |
| 参考模型 | Reference Model | RLHF/DPO 中冻结的 SFT 模型快照，用于 KL 约束新模型不偏离原行为 | ch10 |
| 人类反馈强化学习 | RLHF (Reinforcement Learning from Human Feedback) | 用人类偏好数据训练 reward model，再用 RL（PPO）优化策略模型 | ch10 |
| 监督微调 | SFT (Supervised Fine-Tuning) | 用 prompt-response 对继续训练预训练模型，让其学会按指令回答 | ch09、ch10 |

## 分布式训练与通信

| 术语 | 英文 | 一句话定义 | 主要章节 |
|------|------|------|------|
| AllGather | AllGather | 集合通信原语：每张卡贡献一份数据，最后所有卡都拿到全量拼接结果 | ch11 |
| AllReduce | AllReduce | 集合通信原语：所有卡的数据按 op（如 sum）聚合后，每张卡都拿到聚合结果 | ch11 |
| 数据并行 | DP / DDP (Distributed Data Parallel) | 每张卡持有完整模型副本，各自处理不同 batch，反向传播后 AllReduce 同步梯度 | ch11 |
| 全分片数据并行 | FSDP (Fully Sharded Data Parallel) | PyTorch 原生的 ZeRO-3 等价实现，参数 / 梯度 / 优化器状态全部分片 | ch11 |
| InfiniBand | InfiniBand | 数据中心高速互联标准，RDMA 直通内存，跨机器训练首选 | ch01、ch11 |
| NCCL | NCCL (NVIDIA Collective Communications Library) | NVIDIA GPU 集合通信库，支持 NVLink / PCIe / InfiniBand，GPU 训练必备 | ch11 |
| 流水线并行 | PP (Pipeline Parallelism) | 按层切分模型放到不同卡，前 N 层在 A 卡，后 M 层在 B 卡，micro-batch 流水线传递 | ch11 |
| ReduceScatter | ReduceScatter | 集合通信原语：先 reduce 再按片段分发，每张卡拿到 1/N 的聚合结果 | ch11 |
| Ring All-Reduce | Ring All-Reduce | N 张卡排成环，依次做 reduce-scatter + all-gather，通信量与 N 无关 | ch11 |
| 张量并行 | TP (Tensor Parallelism) | 同一层内按 head/column 切分到不同卡，每步需要 AllReduce 同步，带宽要求高 | ch05、ch11 |
| ZeRO | ZeRO (Zero Redundancy Optimizer) | DeepSpeed 提出的训练显存优化方案，Stage 1/2/3 分别分片优化器状态/梯度/参数 | ch11 |

## 部署与可观测

| 术语 | 英文 | 一句话定义 | 主要章节 |
|------|------|------|------|
| ClusterIP | ClusterIP | K8s Service 默认类型，集群内 IP，集群外不可访问 | ch12 |
| GPU Operator | NVIDIA GPU Operator | 一站式管理 GPU driver、container toolkit、device plugin、DCGM exporter 的 K8s Operator | ch12 |
| Helm | Helm | K8s 应用包管理器，把多个 manifest 模板化为 Chart 便于复用和版本管理 | ch12 |
| HPA | HPA (Horizontal Pod Autoscaler) | K8s 原生自动扩缩容控制器，按指标和目标值动态调整 Pod 副本数 | ch12、ch13 |
| Ingress | Ingress | K8s 的 L7 路由层，用域名 / 路径把外部请求转发到内部 Service | ch12 |
| KEDA | KEDA (Kubernetes Event-Driven Autoscaling) | 基于事件源（队列长度、Prometheus 指标）的扩缩容 Operator，HPA 的增强 | ch12、ch13 |
| LoadBalancer | LoadBalancer | K8s Service 类型，云厂商分配公网 IP 直接暴露服务 | ch12 |
| Langfuse | Langfuse | 开源可自部署的 LLM 可观测平台，支持 prompt 追踪 / token 统计 / 成本分析 | ch13 |
| Loki | Loki | Grafana 出品的轻量日志聚合系统，使用 label 索引按时间序列存储 | ch13 |
| NodePort | NodePort | K8s Service 类型，在每个 Node 上开放固定端口，开发测试用 | ch12 |
| OTel | OpenTelemetry | 厂商中立的可观测数据采集规范，包含 Traces / Metrics / Logs 三类 | ch13 |
| Prometheus | Prometheus | CNCF 时序数据库，主动 pull 模式抓取 metrics，K8s 可观测事实标准 | ch13 |
| Tempo | Tempo | Grafana 出品的分布式追踪后端，存储 OTel/Jaeger 格式的 trace | ch13 |

## RAG 与检索

| 术语 | 英文 | 一句话定义 | 主要章节 |
|------|------|------|------|
| 近似最近邻 | ANN (Approximate Nearest Neighbor) | 在接受极小精度损失的前提下用索引结构大幅加速相似向量查找 | ch14 |
| BM25 | BM25 | 经典稀疏检索算法，基于词频和文档长度归一化打分，精确匹配场景强 | ch14 |
| 切块 | Chunking | 把长文档切成适合 Embedding 模型输入长度的片段 | ch14 |
| ColPali / ColQwen | ColPali / ColQwen | 直接对 PDF 页面图像做 late interaction 检索的多模态模型，绕开 OCR | ch15 |
| Cross-Encoder | Cross-Encoder | Reranker 用的模型架构，把 query 和 doc 拼接后一起喂入做全局 attention，准但慢 | ch14 |
| Bi-Encoder | Bi-Encoder | 普通 embedding 模型架构，把 query 和 doc 分别编码后做点积，快但不如 Cross-Encoder 准 | ch14 |
| 稠密检索 | Dense Retrieval | 用 Embedding 向量做相似度检索，理解语义但不擅长精确匹配 | ch14 |
| 混合检索 | Hybrid Search | 同时跑稠密（向量）和稀疏（BM25）检索，用 RRF 等算法融合结果 | ch14 |
| HNSW | HNSW (Hierarchical Navigable Small World) | 图索引算法，查询快但内存占用大，适合数据量 < 5000 万的场景 | ch14 |
| IVF | IVF (Inverted File Index) | 量化 + 倒排索引，内存友好但需要训练，适合中大规模数据 | ch14 |
| Matryoshka | Matryoshka Embedding | 训练时对子向量也施加损失，使得截断维度后仍保持检索精度 | ch14 |
| MCP | MCP (Model Context Protocol) | Anthropic 提出的标准协议，统一 Agent 和外部数据源 / 工具的接入方式 | ch14 |
| MTEB | MTEB (Massive Text Embedding Benchmark) | Embedding 模型的标准评测基准，按任务平均分排序 | ch14 |
| 检索增强生成 | RAG (Retrieval-Augmented Generation) | 生成前从知识库检索相关片段拼入 prompt，减少幻觉，扩展知识边界 | ch14 |
| 重排序 | Reranker / Reranking | 用 Cross-Encoder 对 Top-K 检索结果重新打分排序，提升精度 | ch14 |
| 稀疏检索 | Sparse Retrieval | BM25 / SPLADE 类基于词权重的稀疏向量检索方法 | ch14 |
| 向量数据库 | Vector Database | 专为高维浮点向量设计的数据库，提供 ANN 检索能力 | ch14 |

## 多模态

| 术语 | 英文 | 一句话定义 | 主要章节 |
|------|------|------|------|
| 自动语音识别 | ASR / STT (Automatic Speech Recognition) | 把语音音频转换为文字 | ch15 |
| CLIP | CLIP (Contrastive Language-Image Pre-Training) | 通过对比学习把图片和文本对齐到同一向量空间，支持图文互检 | ch15 |
| 投影层 | Projection Layer | VLM 架构中把 ViT 输出维度对齐到 LLM Embedding 维度的线性 / MLP 层 | ch15 |
| 实时率 | RTF (Real-Time Factor) | 合成或处理 1 秒音频所需的计算时间，RTF < 1 表示可实时处理 | ch15 |
| 语音合成 | TTS (Text-to-Speech) | 把文字转换为语音音频 | ch15 |
| 视觉 Transformer | ViT (Vision Transformer) | 把图片切成 patch 后用 Transformer 编码的视觉模型，是 VLM 的视觉端主流 | ch15 |
| 视觉语言模型 | VLM (Vision-Language Model) | 同时接受图片和文本输入的语言模型，如 Qwen2-VL / LLaVA | ch15 |
| Whisper | Whisper | OpenAI 开源的多语言 ASR 模型，中文场景可考虑 FunASR / SenseVoice | ch15 |

## 硬件与底层

| 术语 | 英文 | 一句话定义 | 主要章节 |
|------|------|------|------|
| CUDA | CUDA | NVIDIA 提出的 GPU 并行计算平台，几乎所有 LLM 框架的事实标准 | ch00、ch03 |
| 算术强度 | Arithmetic Intensity | 每字节内存搬运对应的浮点运算次数，决定 kernel 是 compute-bound 还是 memory-bound | ch03 |
| 高带宽显存 | HBM (High Bandwidth Memory) | GPU 主显存，比 GDDR 带宽高一个量级，A100/H100 用 HBM2e/HBM3 | ch01、ch03 |
| MPS | MPS (Metal Performance Shaders) | Apple Silicon 的 GPU 加速后端，对应 PyTorch 的 `mps` device | ch00 |
| NVLink | NVLink | NVIDIA 单机 GPU 间高速互联，H100 NVLink 4 带宽 900 GB/s，TP 推荐的硬件 | ch01、ch11 |
| ROCm | ROCm | AMD GPU 的 CUDA 等价物，开源但生态远不如 CUDA | ch01 |
| 流式多处理器 | SM (Streaming Multiprocessor) | NVIDIA GPU 的核心执行单元，A100 有 108 个 SM，每个 SM 有自己的 Shared Memory | ch03 |
| 张量核心 | Tensor Core | NVIDIA GPU 内专门做矩阵乘加的硬件单元，吞吐远高于通用 CUDA Core | ch03 |
| Warp | Warp | 32 个 Thread 组成的最小调度单位，SM 内同步执行同一条指令（SIMT） | ch03 |
| 显存 | VRAM (Video RAM) | GPU 的内存，LLM 推理 / 训练的核心瓶颈，通常指 HBM | ch01、ch03 |


---

> 本附录来自《LLM Infra 从入门到实践》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-llm-infra](https://github.com/diguike/book-llm-infra)
