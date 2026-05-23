# 第 5 章 vLLM：工业级推理引擎深度剖析

vLLM 目前是开源 LLM 推理引擎中部署最广泛的一个。从 2023 年 UC Berkeley 的一篇论文起步，到现在（截至 2026 年初为 v0.19+），已经是多数公司跑 LLM 服务的默认选择。项目地址：[GitHub](https://github.com/vllm-project/vllm) · [官方文档](https://docs.vllm.ai)。这一章讲清楚它为什么快、怎么用、怎么调。

> **本章操作只在 Linux GPU 服务器执行。** vLLM 不支持 macOS（依赖 CUDA，NVIDIA GPU 通用并行计算平台，PyTorch 等深度学习框架都在它之上跑核），Mac 本地想做接口验证可以用第 6 章的 Ollama 代替，API 形态一致。

## 5.1 为什么需要专门的推理引擎

用 HuggingFace Transformers（HuggingFace 出品的 Python 模型库，提供统一的模型加载 / 推理 / 训练接口，类似 LLM 领域的 lodash + axios 合体）的 `model.generate()` 跑推理，能用，但上不了生产。

### 朴素推理的问题

```python
# 最朴素的推理方式
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
```

看起来很简单，但有三个致命问题：

**1. KV Cache 显存浪费严重**

KV Cache（Key-Value Cache，自回归生成时把每一层 attention 算出来的 K、V 向量缓存下来，避免每生成一个 token 都重算历史）的概念已在第 3 章细讲，这里只看它在服务化场景下的麻烦。HF generate 为每个请求预分配最大长度的 KV Cache。如果 `max_length=4096`，即使实际只生成了 50 个 token，也占着 4096 个 token 的显存。对一个 7B 模型，一个请求的 KV Cache 在 4096 长度时约需 1.6 GB 显存。

KV Cache 的容量公式很直接：

```
KV Cache = 2 (K + V) × num_layers × num_heads × head_dim × seq_len × bytes_per_element
```

以 Qwen2-7B 为例（28 层 × 28 头 × head_dim 128，FP16）：

```
2 × 28 × 28 × 128 × 4096 × 2 bytes ≈ 1.6 GB
```

80 GB 的 A100（NVIDIA 上一代旗舰数据中心 GPU，80GB HBM2e 显存）扣掉 14 GB 权重之后，剩下的显存按 1.6 GB/请求算最多同时只能容纳几十个请求。

**2. 静态 Batching 吞吐量低**

Batching（批处理，把多个请求合并成一次 GPU 计算来摊薄启动开销）有静态和动态两种做法。HF generate 用 static batching（静态批处理）：一个 batch 里的所有请求必须等最长的那个生成完，才能开始下一个 batch。短请求被长请求拖累。

```
Static Batching:
请求 A: ████████████░░░░░░░░  (12 token, 等到 20 token 的位置才释放)
请求 B: ████████████████████  (20 token)
请求 C: ██████░░░░░░░░░░░░░░  (6 token, 浪费 14 个位置的计算)
         ────────────────────
         所有请求必须等 B 结束
```

**3. 无法高并发**

没有请求队列、没有异步调度、没有动态 batch。100 个并发请求打过来，要么 OOM（Out Of Memory，显存耗尽，CUDA kernel 直接抛错），要么排队一个一个处理。

### 吞吐量差异

实测数据（Llama-3-8B，A100-80GB，输入 256 token，输出 256 token）：

| 方案 | 吞吐量 (requests/s) | 并发请求数 | 显存利用率 |
|------|---------------------|-----------|-----------|
| HF generate（batch=1） | ~2-3 | 1 | ~25% |
| HF generate（batch=8） | ~8-12 | 8 | ~60% |
| vLLM | ~40-60 | 256+ | ~90% |

vLLM 在高并发场景下的吞吐量是 HF generate 的 **5-15 倍**。

## 5.2 PagedAttention

**PagedAttention（分页注意力）**：vLLM 的核心创新，把 KV Cache 切成固定大小的「页」（block），按需分配、不要求物理连续，从而把传统连续分配方案 60-80% 的显存浪费压到 4% 以下。思路完全是借操作系统的虚拟内存（virtual memory）和页表（page table）那套搬过来的——前端开发者可以把它类比成浏览器把一个长列表用「虚拟滚动 + 分块缓存」管理起来。论文：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（Kwon et al., SOSP 2023）。

### 传统 KV Cache 的问题

KV Cache 存储每一层、每个 attention head（注意力头，多头注意力中并行计算的一个子空间）的 Key 和 Value 向量。随着序列增长，KV Cache 线性增长。

传统方案的做法：为每个请求预分配一块连续的显存，大小等于 max_sequence_length。

```
GPU 显存:
┌────────────────────────────────────────────────────────┐
│ 请求 A 的 KV Cache [████████░░░░░░░░░░░░]  50% 浪费   │
│ 请求 B 的 KV Cache [██░░░░░░░░░░░░░░░░░░]  90% 浪费   │
│ 请求 C 的 KV Cache [████████████████░░░░]  20% 浪费   │
│ ░░░░░░░░░░░░░░░░░░░  无法分配新请求（碎片化）         │
└────────────────────────────────────────────────────────┘
```

实际测量：传统方案中 KV Cache 的有效利用率只有 20-40%。超过一半的显存被浪费在"预留但未使用"的空间上。

### 借鉴操作系统的虚拟内存

PagedAttention 的核心思想：**把 KV Cache 拆成固定大小的"页"（block，KV Cache 在物理显存里的最小分配单元，默认存 16 个 token 的 K/V），按需分配，不要求物理连续。**

```
逻辑视图（每个请求看到的连续 KV Cache）:
请求 A: [Block 0][Block 1][Block 2][Block 3]

物理视图（GPU 显存中的实际存储，不连续）:
┌──────────────────────────────────────────────┐
│ [A-B2] [B-B0] [A-B0] [C-B1] [A-B3] [B-B1]  │
│ [C-B0] [A-B1] [Free] [Free] [C-B2] [Free]   │
└──────────────────────────────────────────────┘

Block Table（block table，逻辑块 → 物理块的映射表，等价于操作系统的页表 page table）:
请求 A: {0→2, 1→7, 2→0, 3→4}
请求 B: {0→1, 1→5}
请求 C: {0→6, 1→3, 2→10}
```

每个 block 存储固定数量的 token（默认 16 个）的 KV 向量。新 token 生成时追加到最后一个 block，block 满了再分配新 block。

### 显存利用率提升

PagedAttention 带来的改进：

- **内部碎片（internal fragmentation，已分配但没用满的空间）**：只有最后一个 block 可能有未填满的空间，浪费 < 4%（传统方案浪费 60-80%）
- **外部碎片（external fragmentation，空闲块零散导致放不下大对象）**：block 是固定大小的，没有外部碎片
- **有效利用率**：从 20-40% 提升到 **>96%**
- **并发量**：同样显存下能同时处理的请求数增加 2-4 倍

还有一个附带好处：**共享前缀的请求可以共享 block。** 比如 100 个请求用同一个 system prompt（系统提示词，对话最开始用来定义模型角色 / 规则的那段固定文本），这些请求的 KV Cache 前面部分指向相同的物理 block，只需要一份存储。这就是 Prefix Caching（前缀缓存，把多个请求共享的 prompt 前缀的 KV Cache 复用一份，避免重复 prefill）的基础。

## 5.3 Continuous Batching

PagedAttention 解决显存问题，Continuous Batching（连续批处理，又叫 in-flight batching / iteration-level batching，每个 decode step 都重新组装一次 batch，请求一完成就立刻让位）解决吞吐量问题。

### Static Batching 的浪费

```
时间 →
Static Batch 1:
  请求 A: ████████ (done)
  请求 B: ████████████████████ (done)
  请求 C: ████ (done, 但要等 B)
  ────────────────────── batch 结束，才能处理新请求

Static Batch 2:
  请求 D: ██████████████ (done)
  请求 E: ██ (done, 等 D)
```

请求 C 在第 4 步就生成完了，但 GPU 上它的"座位"空着，直到 B 生成完。

### Continuous Batching

请求完成后立刻退出 batch，空出的位置立刻被等待中的新请求填入。

```
时间 →
Step  1: [A][B][C]     ← 三个请求同时处理
Step  2: [A][B][C]
Step  3: [A][B][C]
Step  4: [A][B][D]     ← C 完成，D 立刻加入
Step  5: [A][B][D]
Step  6: [E][B][D]     ← A 完成，E 立刻加入
Step  7: [E][B][D]
Step  8: [E][F][D]     ← B 完成，F 立刻加入
...
```

GPU 始终在满负荷处理请求，没有"等待"的浪费。

实际效果：同样硬件下，Continuous Batching 相比 Static Batching 可以提升 **2-5x** 的吞吐量。长短请求混合的场景下差距更大。

vLLM 的调度器（scheduler，每一步决定哪些请求进 batch、给谁分配 block、要不要抢占别人的中央组件）在每个 decode step 都会检查：
1. 有没有请求完成了？释放其 KV Cache block
2. 有没有等待中的请求？为其分配 block，加入 batch
3. 当前显存够不够？不够就 preempt（preemption，抢占，OS 调度借词，把一个跑得动的请求强行从 batch 里踢出去让出资源）低优先级请求——把它的 KV Cache swap（swapping，把显存里的 KV Cache 暂时换到 CPU 内存，等需要时再换回来，等价于 OS 的页面换入换出）到 CPU 内存，等显存空出来再换回来继续 decode；如果 CPU 内存也满了，就直接丢弃该请求的 KV Cache，后续重新跑一遍 prefill 重建状态

## 5.4 部署实战

### 安装

```bash
# 推荐用 pip，需要 CUDA 12.1+
pip install vllm

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

vLLM 对环境要求：
- Python 3.9+
- CUDA 12.1+（推荐 12.4+）
- GPU 算力 7.0+（V100 及以上；GPU 算力 / compute capability 是 NVIDIA 给每代架构编的版本号，7.0 = Volta，8.0 = Ampere，9.0 = Hopper）
- 足够的 GPU 显存（模型大小 + KV Cache）

### 国内模型下载

国内直接从 HuggingFace（HuggingFace Hub，全球最大的开源模型 / 数据集托管平台，相当于 LLM 领域的 npm registry）下载模型速度很慢，推荐两种方案：

**方案 1: HF 镜像**
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2-7B --local-dir ./models/Qwen2-7B
```

**方案 2: ModelScope（阿里出品的国内模型平台，等同国内版 HuggingFace Hub）**
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2-7B', cache_dir='./models')"
```

下载完后启动 vLLM 时指向本地路径：
```bash
python -m vllm.entrypoints.openai.api_server --model ./models/Qwen2-7B
```

### 单卡部署 Qwen2-7B

```bash
# 启动 OpenAI 兼容的 API 服务（OpenAI Compatible API：复刻 OpenAI /v1 端点的协议，让任何 OpenAI SDK 都能直接对接 vLLM）
vllm serve Qwen/Qwen2-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

模型会自动从 Hugging Face 下载。首次启动需要几分钟，后续启动约 30-60 秒。

启动成功后：

```bash
# 测试
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2-7B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }'
```

### 多卡部署：Tensor Parallelism

模型太大放不进单卡？用 tensor parallelism（TP，张量并行，把每一层的权重矩阵按 head 或列切到多卡，每步算完做一次 all-reduce 同步；第 3 章在显存预算里提过 TP，第 11 章会讲实现细节）把模型切到多张卡上。

```bash
# 2 卡部署 —— 每张卡装一半模型
vllm serve Qwen/Qwen2-72B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

# 4 卡部署
vllm serve Qwen/Qwen2-72B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 8192

# 跨机器部署（8 卡 × 2 机器）
# TP=8（每机器内），PP=2（跨机器）
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2
```

Tensor Parallelism 的经验法则：
- `tensor-parallel-size` 必须能整除模型的 attention head 数
- 通常设为单机 GPU 数量（2/4/8）
- 跨机器用 Pipeline Parallelism（PP，流水线并行，按层把模型切到多机），因为 TP 需要高带宽的 NVLink

补一个 TP 和 PP 的区别：TP 在**同一层内**把权重按 head 或 column 切到多卡，每一步都要做 all-reduce（集合通信原语，多卡把各自的部分结果汇总成全局结果再分发回去）同步中间结果，对卡间带宽要求高（所以一般只在 NVLink 同机内用）；PP 是按**层**切——前 N 层放机器 A，后 M 层放机器 B，机器之间像流水线一样传递激活值（activation，神经网络一层的中间输出），每步只有一次小数据量的点对点通信，跨机器跑得动。

### 关键启动参数

```bash
vllm serve <model> \
    # 基础配置
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name my-model \        # API 中的模型名称
    --api-key sk-xxx \                    # 设置 API key

    # 性能参数
    --max-model-len 8192 \                # 最大上下文长度
    --gpu-memory-utilization 0.90 \       # GPU 显存利用率上限
    --max-num-seqs 256 \                  # 最大并发请求数
    --max-num-batched-tokens 8192 \       # 一次 prefill 的最大 token 数

    # 量化
    --quantization awq \                  # 使用 AWQ（Activation-aware Weight Quantization，激活感知权重量化，主流 4bit 量化算法之一，第 7 章细讲）量化模型
    --dtype auto \                        # 数据类型，auto 会自动选择

    # 并行
    --tensor-parallel-size 2 \            # Tensor 并行度
    --pipeline-parallel-size 1 \          # Pipeline 并行度

    # 高级优化
    --enable-prefix-caching \             # 开启 Prefix Caching
    --enable-chunked-prefill              # 开启 Chunked Prefill（把长 prefill 切片，和 decode 交织执行）
```

## 5.5 性能调优

### gpu-memory-utilization

控制 vLLM 使用多少比例的 GPU 显存。默认 0.9（90%）。

```
GPU 显存分配:
┌──────────────────────────────────────┐
│  模型权重 (固定)              ~40%   │
├──────────────────────────────────────┤
│  KV Cache (动态)              ~50%   │  ← gpu-memory-utilization 控制的部分
├──────────────────────────────────────┤
│  预留 (CUDA context + 碎片)   ~10%   │  ← CUDA context: 每个进程绑定到 GPU 上的运行时上下文，存放 stream、module 等
└──────────────────────────────────────┘
```

- `0.9`：默认值，适合大多数场景
- `0.95`：激进，显存紧张时可以尝试，但可能 OOM
- `0.7-0.8`：保守，适合同一张卡还要跑其他任务时

KV Cache 越大 → 能同时处理的请求越多 → 吞吐量越高。

### max-model-len

限制模型能处理的最大序列长度（输入 + 输出）。

```bash
# 模型原始支持 32768，但实际业务用不到那么长
# 降低 max-model-len 可以减少 KV Cache 预分配，腾出显存给更多并发
vllm serve Qwen/Qwen2-7B-Instruct --max-model-len 4096
```

设置策略：
- 分析实际请求的 token 长度分布，取 P99（P99 延迟 / 长度，统计学的 99 百分位，意思是 99% 的请求都小于这个值）作为 max-model-len
- 比如 99% 的请求都在 4096 token 以内，就设 4096 而不是默认的 32768
- 减少 max-model-len 从 32K 到 4K，同等显存下并发量可以提升 4-8 倍

### Prefix Caching

多个请求共享相同前缀时（相同的 system prompt），可以复用 KV Cache。

```bash
vllm serve Qwen/Qwen2-7B-Instruct --enable-prefix-caching
```

典型场景：
- 所有请求带同一个 system prompt（Agent 场景很常见）
- 多轮对话中前面的轮次相同
- RAG（检索增强生成，先从知识库捞相关文档再喂给模型，已在 ch01 介绍）场景中多个请求查询相同的文档片段

效果：对于有长 system prompt 的 Agent 应用，Prefix Caching 可以降低 TTFT 30-70%。

> **TTFT（Time to First Token）**：从客户端发出请求到收到第一个输出 token 的时间，等于一次完整的 prefill 耗时 + 调度排队。它决定用户感知的"响应速度"，流式输出场景下尤其关键。第 8 章会一起把 TTFT、TPOT、TPS 三个指标讲完。

### Chunked Prefill

**Chunked Prefill（分块预填充）**：把长 prompt 的 prefill 拆成多个小块（chunk），和其他请求的 decode 交替执行，避免一个超长 prefill 阻塞其他请求的 decode。

```bash
vllm serve Qwen/Qwen2-7B-Instruct --enable-chunked-prefill
```

每个 chunk 的大小由 `--max-num-batched-tokens` 控制（默认 8192）。一个超过这个值的 prefill 会被切成多步，每步和当前 batch 里其他请求的 decode 交织执行。切分不会破坏 KV Cache 的完整性：每个 chunk 算完都会把结果追加到对应请求的 block table 上，下一个 chunk 进来时能直接看到前面已经计算好的 K/V，不需要重做。

在混合长短请求的场景下，Chunked Prefill 显著改善短请求的延迟，避免被长请求"卡住"。

## 5.6 OpenAI Compatible API

vLLM 提供完全兼容 OpenAI API 的接口，这意味着你已有的代码几乎不用改。

### 支持的接口

```
POST /v1/chat/completions    ← 对话补全
POST /v1/completions         ← 文本补全
POST /v1/embeddings          ← 文本嵌入
GET  /v1/models              ← 模型列表
```

### 直接替换 base_url

```python
from openai import OpenAI  # OpenAI 官方 Python SDK，等价于 npm 包 openai 的 Python 版

# 原来调 OpenAI
# client = OpenAI(api_key="sk-xxx")

# 改成调 vLLM，只需要改 base_url
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM 默认不需要 key
)

response = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain PagedAttention in 3 sentences."},
    ],
    temperature=0.7,  # temperature: 采样温度，越高输出越发散，0 等同贪心解码
    max_tokens=200,
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

### 对接 Agent 应用

对于用 OpenAI SDK 构建的 Agent 应用，切换到 vLLM 只需要：

```typescript
// TypeScript / Node.js
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://your-vllm-server:8000/v1',
  apiKey: 'not-needed',
});

// 后面的代码完全不用改
const response = await client.chat.completions.create({
  model: 'Qwen/Qwen2-7B-Instruct',
  messages: [{ role: 'user', content: 'Hello' }],
});
```

vLLM 支持的 OpenAI 兼容特性：
- `stream: true` — Streaming（流式）输出，token 一边生成一边通过 SSE 推回客户端
- `tools` — 函数调用 / 工具调用
- `response_format: { type: "json_object" }` — JSON 模式（强制输出合法 JSON 的约束解码模式）
- `logprobs` — 返回 token 的 log 概率（每个候选 token 的对数概率值，常用于调试和评估）
- `n` — 一次生成多个回复
- `stop` — 自定义停止词

不兼容的地方：
- `seed` 参数（随机种子，理论上锁定后输出可复现）在分布式推理时不保证完全确定性
- 部分模型的 tool calling 格式可能与 OpenAI 有细微差异
- 不支持 OpenAI 特有的 `gpt-4-vision-preview` 等模型名

### Tool Calling

Tool Calling（工具调用 / 函数调用，模型按约定 schema 输出一段 JSON，外部代码据此调用真实工具——Agent 系统能跑起来的基础）是 Agent 工程师最关心的功能。vLLM 支持 OpenAI 格式的 tool calling，前提是模型本身支持（Qwen2、Llama 3.1+ 等）：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="na")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto",
)

# 模型会返回 tool_calls
message = response.choices[0].message
if message.tool_calls:
    call = message.tool_calls[0]
    print(f"调用工具: {call.function.name}")
    print(f"参数: {call.function.arguments}")
```

启动 vLLM 时需要指定 tool calling 模式：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

`--tool-call-parser` 的选择取决于模型——`hermes` 指 NousResearch Hermes 格式（Qwen2 默认采用这套），`llama3_json` 对应 Meta Llama 3.1 的格式。完整支持列表见 [vLLM Tool Calling 文档](https://docs.vllm.ai/en/latest/features/tool_calling.html)。

---

**本章小结**

vLLM 通过 PagedAttention 解决显存碎片问题，通过 Continuous Batching 解决吞吐量问题，通过 OpenAI 兼容 API 解决接入成本问题。这三个特性是它成为行业标准的关键。生产部署时，最重要的调优参数是 `max-model-len`（根据实际业务设置）和 `enable-prefix-caching`（Agent 场景必开）。

> 示例代码：`examples/ch05-vllm/`


---

> 本章来自《LLM Infra 从入门到实践》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-llm-infra](https://github.com/diguike/book-llm-infra)
