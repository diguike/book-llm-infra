# 第 6 章 推理引擎对比与选型

vLLM 不是唯一选择。不同场景、不同硬件、不同需求下，最优解不同。这一章把市面上主要的推理引擎都过一遍，给出选型建议。

本章涉及的几个主流项目地址先列在这里，方便对照查阅（每个项目后面一句话定位，正文里会再展开）：

- vLLM — <https://github.com/vllm-project/vllm>（第 5 章主角，开源推理引擎事实标准，PagedAttention 的发源地）
- TensorRT-LLM — <https://github.com/NVIDIA/TensorRT-LLM>（NVIDIA 官方推理引擎，靠把模型编译成定制 CUDA kernel 压榨性能）
- SGLang — <https://github.com/sgl-project/sglang>（UC Berkeley 出的推理引擎，主打 Agent / 结构化生成场景）
- llama.cpp — <https://github.com/ggml-org/llama.cpp>（纯 C/C++ 写的开源 LLM 推理库，CPU 也能跑）
- Ollama — <https://github.com/ollama/ollama>（基于 llama.cpp 的本地一键运行工具，类似 LLM 版的 Docker Desktop）
- MLC-LLM — <https://github.com/mlc-ai/mlc-llm>（Machine Learning Compilation，机器学习编译方案，把模型编译到 CUDA/Metal/WebGPU 等多个后端）
- Text Generation Inference (TGI) — <https://github.com/huggingface/text-generation-inference>（HuggingFace 官方推理引擎，Rust 实现的后端）

> HuggingFace（HF）：全球最大的开源模型托管平台，类比 GitHub 之于代码。本章后面提到「HF 生态」「HF 模型仓库」都是指它。

## 6.1 TensorRT-LLM

NVIDIA（GPU 厂商，CUDA 生态的所有者）的亲儿子。如果你全是 NVIDIA GPU 且追求极致性能，它是性能天花板。

### 核心思路：编译优化

TensorRT-LLM 不是直接执行 PyTorch 模型，而是把模型编译成高度优化的 CUDA kernel（CUDA kernel 指运行在 GPU 上的并行计算函数，类比 CPU 上的一个 C 函数，但成千上万个线程同时跑同一段代码）。类似 C++ 代码编译成机器码，运行时没有解释开销。

```
PyTorch 模型
    │
    ▼
TensorRT-LLM 编译
    │  - 算子融合（多个小算子合成一个大 kernel）
    │  - 精度优化（FP8 / NVFP4 自动混合精度）
    │  - 内存优化（减少中间结果的显存占用）
    ▼
TensorRT Engine（二进制文件）
    │
    ▼
高效推理（比 vLLM 快 10-30%）
```

### 关键特性

- **FP8 / NVFP4 量化**（FP8 = 8-bit 浮点格式，NVFP4 = NVIDIA 提出的 4-bit 浮点格式，参数位宽更低 → 显存占用更少、算得更快；详见第 7 章）：Hopper 架构（NVIDIA 2022 年发布的数据中心 GPU 架构，代表型号 H100/H200）支持硬件级 FP8，Blackwell 架构（Hopper 的下一代，2024 年发布）支持 NVFP4。几乎无质量损失，推理速度翻倍
- **Speculative Decoding**（投机解码，先用一个小的 draft 模型快速猜出几个 token，再由大模型一次验证多个，相当于"批量预测+一次验证"加速 decode）：支持 EAGLE-3 等先进算法，吞吐量提升可达 3.6x。EAGLE-3 是投机解码的改进版，把额外的 draft 模型换成在 target 模型上挂的轻量"草稿头"（draft head），显存开销很小（第 8 章会详细讲投机解码原理）
- **In-flight Batching**（飞行中批处理，TensorRT-LLM 对 Continuous Batching 的叫法；Continuous Batching = 连续批处理，请求一来就动态拼入正在跑的 batch，类比 Node.js 事件循环里 push 进 microtask 队列，不用等当前 batch 全部跑完）
- **KV Cache 复用**（KV Cache 见 ch04，多请求共享同一段前缀的缓存，避免重复计算）：类似 vLLM 的 Prefix Caching

### 性能数字

Llama-3.1-8B 在 H100 上的 benchmark（batch size = 一次推理同时处理多少个请求 / 序列，3840 这种大 batch 主要是离线压测场景）：
- 吞吐量（throughput，单位时间整个服务能产出多少 token，衡量"系统总产能"）：~11,000 tokens/s
- TPOT（Time Per Output Token，每生成一个 token 的平均耗时，衡量 decode 阶段的延迟）：~7.3ms
- 比 vLLM 快 10-30%（具体取决于模型和配置）

> 数据来自 [NVIDIA TensorRT-LLM 官方 perf-overview](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-overview.md)（v1.0，2025 Q1 公布）。社区第三方 benchmark 的数字会有差异，部署前以自己业务的 prompt 分布实测为准。

### 痛点

1. **只支持 NVIDIA GPU**：AMD、Intel、Apple Silicon 全部不行
2. **编译过程复杂**：模型需要先转换成 TensorRT 格式，编译可能花几十分钟到几小时
3. **版本迭代慢**：新模型架构支持通常比 vLLM 晚几周
4. **生态封闭**：与 NVIDIA 的 Triton Inference Server（NVIDIA 的开源模型服务框架，统一调度 TensorRT-LLM/PyTorch/ONNX 等多种后端；注意它和 OpenAI Triton 编译器只是重名，没关系）深度绑定

### 适用场景

大规模 GPU 集群、追求极致性能、有专人维护 infra 的团队。不适合快速迭代和开发调试。

## 6.2 SGLang

UC Berkeley（加州大学伯克利分校，硅谷腹地的顶级 CS 院校，vLLM 也出自这里）LMSYS 团队（LMSYS = Large Model Systems Org，伯克利的开源大模型组织，Chatbot Arena 排行榜也是他们做的）的作品，[GitHub](https://github.com/sgl-project/sglang) · [文档](https://docs.sglang.ai)。定位：比 vLLM 更适合 Agent 和结构化生成的场景。

### RadixAttention

SGLang 的核心创新。用 Radix Tree（基数树，一种压缩前缀树数据结构，能高效查找/插入有公共前缀的字符串，类比前端路由库里的 trie router）管理 KV Cache 的前缀复用。

```
Radix Tree 示例:

                    [system prompt]
                    /              \
          [用户问题 A]           [用户问题 B]
          /         \                 |
    [追问 A1]   [追问 A2]       [追问 B1]
```

与 vLLM 的 Prefix Caching 的区别：
- vLLM 的 Prefix Caching 基于精确的 hash 匹配，前缀必须完全相同
- RadixAttention 用树结构管理，自动发现和复用最长公共前缀
- 多轮对话场景下，RadixAttention 的缓存命中率更高（75-95%）

实际效果：对于 Agent 场景（固定 system prompt + tools 定义 + 多轮调用），TTFT（Time To First Token，首个 token 的延迟，见 ch04）降低 30-60%。

### 结构化生成（JSON Mode）

JSON Mode（结构化输出模式，让模型保证只产出合法 JSON 的功能，类比 OpenAI API 里的 `response_format: { type: "json_object" }`）。Agent 经常需要模型输出结构化 JSON。SGLang 在这方面做了深度优化。

```python
import sglang as sgl

@sgl.function
def extract_info(s, text):
    s += "Extract information from: " + text + "\n"
    s += "Output JSON:\n"
    s += sgl.gen("result",
                 regex=r'\{"name": "[^"]+", "age": \d+\}')
```

SGLang 的结构化生成比 vLLM 的 `guided_decoding`（受约束解码，运行时在每一步采样前过滤掉不合法的 token，强制输出符合 JSON Schema 或正则的内容）快得多，原因是它在 token 级别做了约束传播优化，不需要在每一步都重新计算整个正则表达式的状态。

### 与 vLLM 的性能对比

SGLang 在以下场景通常比 vLLM 更快：
- 多轮对话（RadixAttention 的优势）
- 结构化输出（JSON/正则约束）
- 高前缀重叠率的工作负载

vLLM 的优势：
- 生态更成熟，社区更大
- 模型支持范围更广
- 文档和教程更完善
- 企业级支持更完善

### 部署

> 以下命令适用于 SGLang v0.4-v0.5。新版本可能微调参数名，最新写法以 [官方安装文档](https://docs.sglang.ai/start/install.html) 为准。

```bash
# 安装
pip install sglang[all]

# 启动服务（同样兼容 OpenAI API）
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-7B-Instruct \
    --port 8000 \
    --tp 1    # tp = Tensor Parallel size，张量并行的卡数，单卡填 1，多卡按卡数填
```

SGLang 同样提供 OpenAI 兼容的 API，切换成本很低。

## 6.3 Ollama / llama.cpp

本地推理的事实标准。开发调试用 Ollama，搞清楚底层用 llama.cpp。

### llama.cpp

Georgi Gerganov（保加利亚开发者，llama.cpp / GGML / Whisper.cpp 等一系列 C/C++ ML 库的作者）在 2023 年 3 月用纯 C/C++ 实现的 LLM 推理库。一个人开始的项目，现在是最活跃的开源 LLM 项目之一。

技术特点：

1. **纯 C/C++**：无 Python 依赖，编译后一个二进制文件搞定。极致的可移植性
2. **CPU 推理优化**：AVX2/AVX-512（Intel/AMD CPU 的高级向量扩展指令集，512 位一次能并行算 16 个 float）/ARM NEON（ARM CPU 的 SIMD 扩展，Apple Silicon 也走它）等 SIMD（Single Instruction Multiple Data，单指令多数据并行，一条指令同时处理一组数）指令集优化，CPU 上也能跑得动
3. **GGUF 格式**（见 ch04，GG 系列工具自定义的模型文件格式，把权重和元数据装在一个文件里）：原生量化支持，Q4_K_M（GGUF 量化类型命名，Q4 = 4-bit 量化，K = K-quants 算法变体，M = medium，详见第 7 章）量化下 7B 模型只需 ~4GB 内存
4. **GPU 加速**：支持 CUDA、Metal（Apple 自家的 GPU 计算 API，Apple Silicon 上要走它才能用上 GPU）、Vulkan（跨厂商的低层级 GPU API，AMD/Intel/NVIDIA/Android GPU 都支持）、SYCL（基于 C++ 的跨平台异构计算标准，Intel GPU 主推它）
5. **跨平台**：Linux、macOS、Windows、Android、iOS、ChromeOS

一个 7B 模型在不同硬件上的 decode（见 ch04，自回归地一个一个生成 token 的阶段）速度（Q4_K_M 量化）：

| 硬件 | TPS（tokens per second，见 ch04，每秒生成 token 数，衡量单请求 decode 速度） | 备注 |
|------|-----|------|
| M2 Max (GPU) | ~80 | Metal 加速 |
| M2 Max (CPU) | ~25 | 纯 CPU |
| RTX 4090 | ~120 | CUDA |
| RTX 3060 12GB | ~55 | CUDA |
| i9-13900K (CPU) | ~15 | AVX2 |
| Raspberry Pi 5 | ~2-3 | ARM NEON |

llama.cpp 也提供了 server 模式，兼容 OpenAI API：

```bash
# 编译
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release

# 启动 server
./build/bin/llama-server \
    -m models/qwen2-7b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99    # 所有层放 GPU
```

### Ollama

llama.cpp 的用户友好封装。把"下载模型 + 量化 + 配置 + 运行"简化成一条命令。命令行体验类比 `docker run`：一个名字搞定一切。

安装方式按机器分两种：

**Mac 本地：**
```bash
# 推荐直接下载官方安装包：https://ollama.com/download
# 装完之后菜单栏会有一个 Ollama 后台应用，自动监听 11434 端口
# 也可以走 Homebrew
brew install ollama
```

**Linux GPU 服务器：**
```bash
curl -fsSL https://ollama.com/install.sh | sh
# 装完后用 systemctl 管理：
sudo systemctl start ollama
```

下载并运行模型的命令两边一致：

```bash
# 运行模型（自动下载 + 启动）
ollama run qwen2:7b

# 作为 API 服务（自动启动后台服务）
curl http://localhost:11434/api/chat -d '{
    "model": "qwen2:7b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
}'
```

Ollama 的最新特性（2025-2026）：
- **桌面应用**：macOS、Windows 原生 GUI，支持拖拽 PDF/图片
- **MLX 加速**：MLX 是 Apple 官方推出的 Apple Silicon 机器学习框架，专门吃满 M 系列芯片的统一内存和 GPU。Apple Silicon 上 prompt 处理速度提升 1.6x，生成速度提升约 2x
- **Thinking Mode**（思考模式，控制推理模型「思考链」是否暴露给客户端，类似 OpenAI o1/Anthropic extended thinking 那种）：支持推理模型的思考链可见性控制
- **结构化输出**：JSON Schema（JSON 模式描述规范，前端写 OpenAPI / TS 的 zod schema 都在表达类似的东西）支持
- **工具调用**（function calling，让模型按格式返回要调用的函数名 + 参数，Agent 框架的核心能力）：Streaming 中的实时函数调用

Ollama 还兼容 OpenAI API（`/v1/chat/completions`），可以直接用 OpenAI SDK 调用。

### 适用场景

- **开发调试**：本地跑个小模型测 prompt、测 Agent 逻辑，不需要 GPU 服务器
- **个人使用**：Mac 上跑 7B 模型做日常助手，体验很好
- **边缘部署**：嵌入式设备、手机、离线场景
- **CI/CD 测试**：在 CPU-only 的 CI 环境中做模型相关的集成测试

**不适合**：高并发的线上服务。Ollama 的并发能力和吞吐量远不如 vLLM/SGLang。

## 6.4 MLC-LLM

机器学习编译（Machine Learning Compilation）方案，陈天奇团队（XGBoost、TVM、MXNet 等多个机器学习基础设施项目的主导者，现任 CMU 助理教授）的作品。

### 核心思路

用 Apache TVM（开源深度学习编译器栈，把高层模型 IR 优化并编译成各硬件平台原生代码，类比 LLVM 之于通用程序语言）编译器把模型编译到不同后端：CUDA、Metal、Vulkan、WebGPU（浏览器里的现代 GPU API，相当于把 Vulkan/Metal/D3D12 在 Web 上做了统一抽象）、OpenCL（早期的跨平台异构并行计算 API，移动 GPU 和部分嵌入式 GPU 还在用）。一次编译，多端运行。

```
PyTorch / HuggingFace 模型
        │
        ▼
   MLC-LLM 编译
        │
   ┌────┼────┬────────┬──────────┐
   ▼    ▼    ▼        ▼          ▼
 CUDA  Metal Vulkan  WebGPU   OpenCL
 (PC)  (Mac) (跨平台) (浏览器)  (移动端)
```

### 跨平台部署

- **iOS / Android**：编译成原生库，集成到 App 中。配合 React Native 可以用 JS API 调用
- **浏览器**：WebLLM（MLC 团队的 web 子项目，把 MLC 编译产物跑在浏览器里）项目，通过 WebGPU 在浏览器中运行 LLM，无需服务器
- **嵌入式**：支持各种 ARM 设备

### 实际表现

MLC-LLM 的性能通常介于 llama.cpp 和 vLLM 之间。在特定硬件（如 Apple Silicon）上，MLC 的编译优化可以比 llama.cpp 快 10-20%。

WebLLM 在浏览器中的表现（Chrome + WebGPU）：
- Llama-3-8B-Q4：~25-35 tokens/s（高端 GPU）
- Phi-3-mini-Q4：~40-50 tokens/s

### 适用场景

- 需要在浏览器中运行 LLM（隐私敏感场景）
- 移动端原生 LLM 集成
- 跨平台统一部署
- 对 TVM 编译栈（即上面提到的 Apache TVM）有经验的团队

### 局限

- 编译过程复杂，文档有限
- 社区比 vLLM/llama.cpp 小得多
- 新模型支持速度较慢
- 生产级部署案例较少

## 6.5 HuggingFace TGI

HuggingFace 官方的推理引擎 [Text Generation Inference](https://github.com/huggingface/text-generation-inference)，Rust 实现的后端（Web server + 推理调度）加 Python 侧的模型代码。和 HF 生态深度集成，支持 FlashAttention（Tri Dao 提出的注意力算子优化实现，通过分块加载、避免反复读写 HBM 来加速 attention，是几乎所有现代推理引擎的标配，第 8 章会详细讲）、Continuous Batching、量化模型（AWQ = Activation-aware Weight Quantization，激活感知的权重量化方法；GPTQ = Generative PTQ，逐层基于 Hessian 信息做权重量化；bitsandbytes，简称 BnB，HuggingFace 生态最常用的 8-bit/4-bit 量化库，三者细节见第 7 章）。如果你的工作流重度依赖 HuggingFace（模型托管、Inference Endpoints — HF 官方的模型托管推理服务，相当于"模型版的 Vercel"），TGI 是最无缝的选择。

### 安装与启动

官方推荐用 Docker 启动，省去 Rust 编译环境：

```bash
# 拉镜像并启动（Qwen2-7B-Instruct 为例）
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id Qwen/Qwen2-7B-Instruct \
    --max-input-length 4096 \
    --max-total-tokens 8192
```

`-v $PWD/data:/data` 把模型缓存挂到宿主机，重启容器不用重新下载。需要登录私有模型时加 `-e HF_TOKEN=...`。

启动好后 TGI 暴露两套 API——原生的 `/generate` 和 OpenAI 兼容的 `/v1/chat/completions`，后者可以直接用 OpenAI SDK 调：

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "tgi",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": false
    }'
```

### 和 vLLM 的几个具体差异

1. **架构语言**：TGI 的调度器和 router 用 Rust 写，启动快、内存占用低；vLLM 是纯 Python，便于二次开发但启动慢一些
2. **KV Cache 管理**：TGI 在 v2 之后也用了 PagedAttention（见 ch01/ch05，仿照操作系统虚拟内存分页管理 KV Cache 的算法）思路，但生态没有 vLLM 那么深——多 GPU 张量并行（Tensor Parallel，把一层网络的矩阵切到多张卡上同时算，第 11 章会详细讲）、Prefix Caching、多模型多 LoRA（Low-Rank Adaptation，低秩适配微调，只训练小的低秩矩阵，原始大模型权重不动，第 9 章详述）这些功能 vLLM 更成熟
3. **HF 生态绑定**：TGI 直接读 HF 模型仓库，支持的模型范围跟着 `transformers` 走；vLLM 自己维护一份模型代码，新模型支持速度往往更快
4. **生产部署**：TGI 是 HuggingFace Inference Endpoints 的官方后端，要把模型部署到 HF 托管平台就只能用它

性能介于 vLLM 和朴素推理之间，社区活跃度不如 vLLM 和 SGLang。新功能（如 LoRA 多适配器、最新的 speculative decoding 算法）的跟进速度也明显慢一拍。

## 6.6 选型指南

### 对比大表

| 特性 | vLLM | TensorRT-LLM | SGLang | llama.cpp/Ollama | MLC-LLM | TGI |
|------|------|---------------|--------|-----------------|---------|-----|
| **性能** | ★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★ | ★★★½ |
| **易用性** | ★★★★ | ★★ | ★★★★ | ★★★★★ | ★★ | ★★★★ |
| **模型支持** | ★★★★★ | ★★★ | ★★★★ | ★★★★ | ★★★ | ★★★★ |
| **GPU 支持** | NVIDIA, AMD | NVIDIA only | NVIDIA, AMD | NVIDIA, AMD, Apple, Intel | 全平台 | NVIDIA, AMD |
| **CPU 推理** | 有限 | 不支持 | 有限 | 优秀 | 有限 | 不支持 |
| **量化** | AWQ, GPTQ, FP8 | FP8, NVFP4, INT4/8 | AWQ, GPTQ, FP8 | GGUF 2-8bit | Q4, Q8 | AWQ, GPTQ, BnB |
| **OpenAI API** | 原生支持 | 通过 Triton | 原生支持 | 原生支持 | 有限 | 兼容 |
| **Streaming** | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| **结构化输出** | 支持 | 有限 | 优秀 | 支持 | 有限 | 支持 |
| **多模态** | 支持 | 支持 | 支持 | 支持 | 有限 | 支持 |
| **社区活跃度** | 极高 | 中等 | 高 | 极高 | 中等 | 中等偏高 |
| **适合团队** | 通用 | 大厂 infra 团队 | Agent 开发者 | 个人/小团队 | 跨平台开发者 | HF 生态用户 |
| **最新版本（截至 2026 年初）** | v0.19+ | v1.0+ | v0.5+ | 持续更新 | v0.1+ | v3.x+ |

### 决策树

```
你的场景是什么？
│
├─ 线上服务（高并发）
│  │
│  ├─ 只有 NVIDIA GPU，追求极致性能
│  │  └─ → TensorRT-LLM
│  │
│  ├─ Agent 场景，大量结构化输出
│  │  └─ → SGLang
│  │
│  └─ 通用场景，需要稳定可靠
│     └─ → vLLM（推荐默认选择）
│
├─ 本地开发/调试
│  │
│  ├─ 快速上手，不想折腾
│  │  └─ → Ollama
│  │
│  └─ 需要定制，了解底层
│     └─ → llama.cpp
│
├─ 边缘/移动端部署
│  │
│  ├─ 手机 App 内嵌 LLM
│  │  └─ → MLC-LLM
│  │
│  ├─ 浏览器端运行
│  │  └─ → WebLLM (MLC)
│  │
│  └─ 嵌入式/IoT
│     └─ → llama.cpp
│
└─ 研究/实验
   └─ → vLLM 或 SGLang（看具体方向）
```

### 实际建议

**大多数团队的最优路径：**

1. **开发阶段**：用 Ollama 在本地跑模型，快速迭代 prompt 和 Agent 逻辑
2. **测试阶段**：用 vLLM 在 GPU 服务器上跑，验证性能和正确性
3. **生产阶段**：vLLM 或 SGLang，根据具体场景选择

**几个具体判断点：**

- 如果你的 Agent 有长 system prompt + 大量工具定义（>2000 token），SGLang 的 RadixAttention 会比 vLLM 的 Prefix Caching 更有优势
- 如果你需要严格的 JSON Schema 输出，SGLang 的结构化生成性能明显更好
- 如果你用 NVIDIA H100/H200 且有专人运维，TensorRT-LLM 的性能值得那些额外的复杂度
- 如果你的用户在端侧（手机/桌面），Ollama 是最省心的分发方案

**避坑提示：**

- 不要在 CPU 上用 vLLM。vLLM 的 CPU 支持是实验性的，性能远不如 llama.cpp
- 不要在单卡上用过大的 tensor-parallel-size。TP=1 时无通信开销，很多时候单卡跑满比 2 卡分拆更快
- 不要忽视 Ollama 的局限性。它适合开发和个人使用，但不要用它扛生产流量
- TensorRT-LLM 的模型转换可能失败或产生精度问题，务必做好回归测试

---

**本章小结**

没有银弹。vLLM 是默认选择，SGLang 在 Agent 场景有优势，TensorRT-LLM 是性能天花板但门槛高，Ollama/llama.cpp 是本地推理标配，MLC-LLM 解决跨平台问题。根据你的实际场景选择，不要过度追求性能而忽视了开发效率和运维复杂度。

> 示例代码：`examples/ch06-engines-comparison/`


---

> 本章来自《LLM Infra 从入门到实践》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)
