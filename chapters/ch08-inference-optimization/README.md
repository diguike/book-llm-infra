# 第 8 章 推理加速技术

上一章解决了"怎么把模型塞进更小的显存"，这一章解决"怎么让模型跑得更快"。

推理加速的本质是在不改变（或极少改变）输出质量的前提下，减少计算量或提高硬件利用率。这里介绍五个核心技术，按照实际工程中的重要程度排序。

## 8.1 FlashAttention

> **FlashAttention**：Tri Dao 等人 2022 年提出的 IO-aware（感知内存读写代价）Attention 实现，通过分块计算把 HBM 读写从 O(n²) 降到 O(n)，是几乎所有现代推理引擎的默认 Attention 后端。第 2 章已经粗略提过它解决"显存占用 + 速度"问题，这一章把内部机制拆开讲。

### 标准 Attention 的问题

回顾第 2 章的 Attention 计算：

```
Q @ K^T → [n, n] 矩阵 → softmax → @ V → 输出
```

这里 Q、K、V 分别是 query / key / value 矩阵（第 2 章已介绍），`softmax` 是把一组实数归一化成概率分布的函数。中间的 `[n, n]` 矩阵是问题的根源。以 Llama 2 7B 单层单头为例：

| Sequence Length | Attention 矩阵大小 | 显存 (FP16) |
|----------------|-------------------|------------|
| 2K | 4M | 8 MB |
| 8K | 64M | 128 MB |
| 32K | 1024M | 2 GB |
| 128K | 16384M | 32 GB |

乘以 32 层 × 32 头，32K context 下光是 Attention 中间矩阵就要 2TB 显存——这么大显然不可能真的全存下来。实际中 PyTorch 会逐步计算释放，但 **HBM**（High Bandwidth Memory，GPU 主显存，第 3 章已介绍）的读写次数依然是 O(n²)。

GPU 的内存层次：

```
┌──────────────┐
│   SRAM       │  每个 SM 约 192KB (A100)
│   (~20 MB)   │  带宽: ~19 TB/s
├──────────────┤
│   HBM        │  80 GB (A100)
│   (显存)     │  带宽: 2 TB/s
├──────────────┤
│   主内存      │  几百 GB
│   (CPU RAM)  │  带宽: ~50 GB/s
└──────────────┘
```

**SRAM**（Static RAM，静态随机存储器，GPU 上挂在每个 SM 内部的片上高速缓存，第 3 章已介绍）比 HBM 快近 10 倍，但小得多。**SM**（Streaming Multiprocessor，流式多处理器，GPU 的基本计算单元）。标准 Attention 的做法是在 HBM 中计算完整的 `[n, n]` 矩阵，来回搬运数据。FlashAttention 的思路：能不能在 SRAM 里分块算完，根本不把完整的 `[n, n]` 写回 HBM？

### FlashAttention 的核心思想

三个关键词：**tiling**（分块计算，把大矩阵切成能装进 SRAM 的小块逐块处理）、**kernel fusion**（算子融合，把多个 GPU kernel 合并成一个执行，省掉中间结果落回 HBM 的开销，类比前端把多个小 HTTP 请求合并成一个 batch 请求）、**recomputation**（重计算，反向传播时重新算前向中间结果而不是存下来）。

```
标准 Attention（多次 HBM 读写）:
  Q, K, V 在 HBM
  → 读 Q, K 到 SRAM, 算 S = Q @ K^T, 写 S 回 HBM
  → 读 S, 算 P = softmax(S), 写 P 回 HBM
  → 读 P, V, 算 O = P @ V, 写 O 回 HBM
  总共 6 次大规模 HBM 读写

FlashAttention（一次搞定）:
  把 Q, K, V 按 block 切分
  → 每次读一小块 Q, K, V 到 SRAM
  → 在 SRAM 中算完局部的 attention + softmax + 输出
  → 用 online softmax 算法把局部结果正确合并
  → 只把最终结果 O 写回 HBM
  HBM 读写次数从 O(n²) 降到 O(n)
```

**online softmax**（在线 softmax）是 FlashAttention 能工作的数学基础——常规 softmax 需要先看完整行才能算最大值做归一化，online softmax 改写成增量公式，允许在不知道全局最大值的情况下逐块计算并最终得到正确结果。这不是近似，结果和标准 Attention 在数值上完全一致（忽略浮点精度差异）。

第三个关键词 **recomputation**（重计算）主要在训练场景生效：前向传播时不把 `[n, n]` 的 attention 矩阵保存到 HBM，反向传播需要它的时候按 block 重新算一遍。用一点额外 **FLOPs**（Floating Point Operations，浮点运算次数，衡量算力消耗的常用单位）换显存，让长序列的训练能跑起来。推理只走前向，这一项默认不开。

### 演进：FlashAttention 1 → 2 → 3

| 版本 | 主要改进 | 相比 v1 加速 |
|------|---------|-------------|
| v1 (2022) | 基础 tiling + online softmax | 基准 |
| v2 (2023) | 优化并行度，减少 non-matmul FLOPs | 2x |
| v3 (2024) | 利用 H100 的异步 TMA 和 FP8 | 1.5-2x over v2 |

表里的 **non-matmul FLOPs** 指除矩阵乘法之外的运算（softmax、mask、缩放等），这些操作在 GPU 上比 matmul 慢得多，v2 重新排布计算顺序压低了它们的占比。**TMA**（Tensor Memory Accelerator，张量内存加速器）是 H100 引入的硬件单元，用异步搬运代替 SM 的同步 load/store；**FP8**（8 位浮点数，第 7 章已介绍）是 Hopper 架构的新数值类型，配合 **FP8 Tensor Core**（专门做低精度矩阵乘的硬件单元）能再翻一倍吞吐。

FlashAttention 2 的关键优化：v1 在 batch 和 head 两个维度并行，v2 额外在 sequence length 维度并行，GPU 利用率从 v1 的 ~50% 提升到 ~70%。

### 实际使用

好消息是你不需要手动调用 FlashAttention。PyTorch 2.0+ 的 `F.scaled_dot_product_attention`（缩放点积注意力，PyTorch 官方的 attention 算子接口，内部会按硬件选择 FlashAttention / Memory-Efficient / 数学版本三种实现之一）默认会选择最优的 attention 实现：

```python
import torch.nn.functional as F

# PyTorch 自动选择 FlashAttention（如果硬件支持）
output = F.scaled_dot_product_attention(query, key, value)
```

vLLM（高吞吐推理引擎，第 5 章已介绍）、**HuggingFace Transformers**（HuggingFace 出品的模型加载与训练库，几乎是 Python 端跑预训练模型的事实标准）等框架内部已经全面使用 FlashAttention。你要做的只是确保 PyTorch 版本 ≥ 2.0，GPU 支持（A100/H100/RTX 3090+）。

实际效果（A100, Llama 2 7B, batch=1）：

| Seq Length | 标准 Attention | FlashAttention 2 | 加速 | 显存节省 |
|-----------|---------------|------------------|------|---------|
| 2K | 12 ms | 5 ms | 2.4x | 4x |
| 8K | 180 ms | 35 ms | 5.1x | 16x |
| 32K | OOM | 420 ms | ∞ | - |

32K 的 case 最能说明问题：标准 Attention 直接 **OOM**（Out Of Memory，显存耗尽），FlashAttention 轻松跑完。

代码示例见 `examples/ch08-inference-optimization/01_flash_attention_demo.py`。

## 8.2 Speculative Decoding

> **Speculative Decoding**（投机解码 / 推测解码）：用一个小模型先"猜"几个 token、再让大模型一次性并行验证的解码加速方法，结果与原模型完全等价，不掉精度。第 1 章曾顺带提过，这一章把它和它的几个变体讲透。

### Decode 的瓶颈

第 2 章讲过，Decode 阶段是 **memory-bound**（内存带宽受限，瓶颈在显存读取速度而不是算力，第 3 章已介绍）：每生成一个 token，GPU 要从显存读取整个模型的权重（7B 模型 = 14 GB @ **FP16**，半精度浮点数），但实际计算量很小（只处理 1 个 token）。GPU 的算力大量闲置，利用率可能不到 10%。

换个角度看：大模型生成 1 个 token 要 30ms，小模型可能只要 5ms。但两者大部分时间都在等显存读取，真正算的时间差别没那么大。

### 核心思想：猜测 + 验证

Speculative Decoding 的思路简单粗暴：

1. 用一个小模型（**draft model**，草稿模型，参数量小、跑得快的辅助模型）快速生成 K 个候选 token（比如 K=5）
2. 把这 K 个 token 一次性送给大模型（**target model**，目标模型，真正要服务的大模型）并行验证
3. 大模型从左到右检查，接受匹配的 token，在第一个不匹配的位置生成正确的 token
4. 丢弃不匹配位置之后的所有候选

```
Draft model (Qwen2-0.5B):
  快速生成: [A, B, C, D, E]            耗时 ~25ms (5 × 5ms)

Target model (Qwen2-7B):
  并行验证: [A, B, C, D, E]            耗时 ~35ms (一次 forward)
  结果:      ✓  ✓  ✓  ✗→[D']
  接受 [A, B, C]，在第一个不匹配的位置直接产出修正 token [D']

收获: 一步得到 4 个 token (3 accepted + 1 corrected)
耗时: 25 + 35 = 60ms
标准方式: 4 × 30 = 120ms
加速: 2x
```

注意被拒绝的位置不是"白跑一趟"——target 模型在这次 **forward**（前向传播，一次模型从输入到输出的完整计算）里同时算出了每个位置的正确预测，第一个被拒绝的草稿 token 会用 target 自己的预测覆盖，所以这一步保底也能拿到 1 个新 token。

为什么验证 5 个 token 和验证 1 个差不多快？因为这 5 个 token 可以像 Prefill（预填充阶段，第 4 章已介绍）一样并行计算——而 Prefill 是 **compute-bound**（算力受限，瓶颈在 GPU 算力而不是显存带宽，第 3 章已介绍），算 5 个和算 1 个的时间差异很小（GPU 算力有富余）。

### 加速比取决于接受率

Draft model 的输出越接近 Target model，接受率越高，加速越明显：

| 接受率 | 预期加速比 (K=5) | 适用场景 |
|--------|-----------------|---------|
| 50% | ~1.3x | 不相关的 draft model |
| 70% | ~1.8x | 同系列小模型 |
| 80% | ~2.2x | 相关度高的 draft model |
| 90% | ~2.8x | 非常匹配，或简单任务 |

选 draft model 的原则：
- **同系列**：Qwen2-0.5B 做 Qwen2-7B 的 draft，效果好
- **不能太大**：draft model 的开销不能超过节省的时间
- **任务相关**：代码生成这类确定性高的任务，接受率更高

### vLLM 中的使用

```bash
# 启动 vLLM，开启 speculative decoding
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B \
    --speculative-model Qwen/Qwen2-0.5B \
    --num-speculative-tokens 5
```

不需要改客户端代码，对外接口完全一样。

vLLM 还支持 **ngram speculation**（也叫 **Prompt Lookup Decoding**，提示查找解码）——不用额外的 draft model，而是从已有的输入/输出中匹配重复的 **n-gram**（连续的 n 个 token 序列）模式来猜测。对于包含重复模式的文本（比如代码、格式化数据），效果不错：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B \
    --speculative-model [ngram] \
    --ngram-prompt-lookup-max 4 \
    --num-speculative-tokens 5
```

代码示例见 `examples/ch08-inference-optimization/02_speculative_decoding.py`。

除了 ngram 和小模型 draft，社区还有几条加速 draft 的工程路线，名字会反复出现在 vLLM/SGLang 的 release note 里：

- **Medusa**：在大模型自己头上接几个轻量的"预测头"（额外的输出线性层），让大模型一次 forward 同时产出多个候选位置的 token，省掉外挂 draft model
- **EAGLE**（Extrapolation Algorithm for Greater Language-model Efficiency）：用大模型的隐藏层特征训练一个轻量草稿网络，比 Medusa 接受率更高，是目前社区最主流的 speculative 变体
- **Lookahead Decoding**：不依赖额外模型，靠 Jacobi 迭代（一种数值迭代解方程的方法）并行预测多个 token，硬解 decode 串行依赖

这些技术在 vLLM 中以不同的 `--speculative-method` 参数暴露，作为 Agent 工程师了解名词、按场景选开关就够。

## 8.3 KV Cache 压缩与管理

第 2 章计算过，Llama 2 7B 在 seq_len=4096、batch=32 时，**KV Cache**（Key-Value Cache，把 Attention 中过去 token 的 key/value 张量缓存下来避免重复计算，第 1、2 章已介绍）就要 64 GB 显存。KV Cache 管理是推理引擎的核心问题之一。

### 量化 KV Cache

最直接的方案：把 KV Cache 从 FP16 量化到 FP8 或 **INT8**（8 位整数，第 7 章已介绍）。

```
FP16 KV Cache: 每个元素 2 bytes
FP8 KV Cache:  每个元素 1 byte  → 显存减半
```

vLLM 支持 FP8 KV Cache：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B \
    --kv-cache-dtype fp8
```

精度损失极小——KV Cache 的数值范围通常比权重更窄，量化友好。多数 **benchmark**（基准测试）显示 FP8 KV Cache 的质量几乎无损。

### Sliding Window Attention

**Sliding Window Attention**（滑动窗口注意力，每个 token 只关注最近 W 个 token，把 Attention 视野限制在固定窗口内）是 Mistral 7B 引入的方案：每个 token 只和最近 W 个 token 做 Attention（W=4096）。

```
标准 Attention (seq_len=32K):
  每个 token 看前面所有 32K token
  KV Cache: 32K × 每 token 大小

Sliding Window (W=4096):
  每个 token 只看前面 4096 token
  KV Cache: 4096 × 每 token 大小（固定）
```

KV Cache 大小不再随 sequence length 增长，而是固定为 W。但代价是长距离依赖能力减弱——超过窗口大小的 token 之间无法直接交互。

实际影响没有理论上那么大，因为信息可以通过多层传递：第 1 层看 4K 窗口，但 32 层叠加后，信息可以传播 32 × 4K = 128K 的有效距离。这类似传话游戏——即使没人能直接和 130K 前的 token 对话，信息可以通过中间层一层一层接力过去。但接力越长信息衰减越严重，实际有效感受野通常比理论上限明显小，对需要"在长文档里精准定位某句话"的任务（典型如大海捞针 benchmark）尤其吃亏。

### StreamingLLM

**StreamingLLM**（流式 LLM 解码方案，MIT 2023 提出，专为无限长输入流设计）更极端：只保留**前几个 token**（**attention sink**，注意力锚点 token，研究发现 Transformer 前几个 token 会被异常高频地 attend，丢掉就崩）和**最近的窗口**。

```
完整 KV Cache:  [t1, t2, t3, t4, ..., t998, t999, t1000]
StreamingLLM:    [t1, t2, t3, t4,  ..., t996, t997, t998, t999, t1000]
                  ↑ sink tokens          ↑ recent window
                  (前 4 个)              (最近 1000 个)
```

研究发现，Transformer 的前几个 token 总是获得异常高的 attention score（即使内容无关），它们充当了"注意力锚点"。丢掉这些 token 会导致输出质量急剧下降，但保留它们 + 最近的窗口就能维持不错的质量。

StreamingLLM 让模型可以处理无限长的输入流（比如实时对话），KV Cache 大小固定，不会 OOM。但它不是万能的——被窗口淘汰的信息就真的丢了。

### H2O (Heavy Hitter Oracle)

**H2O**（Heavy Hitter Oracle，重要击中预言机；"heavy hitter"在统计领域指数据流中出现频率最高的少数元素）是更精细的淘汰策略：不是简单地按位置淘汰，而是追踪每个 token 的累积 **attention score**（注意力得分，softmax 之后的权重），淘汰分数最低的 token。

直觉：有些 token 很重要（被频繁 attend），有些是"填充词"。与其均匀保留最近的窗口，不如保留最重要的 token。

H2O 在保留相同数量 KV 的情况下，比固定窗口的质量更好。但实现更复杂，需要额外维护 attention score 的统计。

## 8.4 Prefix Caching

> **Prefix Caching**（前缀缓存）：识别多个请求共享的前缀 token，复用前缀对应的 KV Cache 而不重新 Prefill。第 1 章曾在 RadixAttention 处提到过，这一章讲 vLLM 这边的工程实现。

### 场景

Agent 应用中，每个请求都带着相同的 **system prompt**（系统提示词，放在对话最前面用于设定模型角色和行为的固定文本）：

```
请求 1: [system_prompt(2000 tokens)] + "帮我搜索天气"(10 tokens)
请求 2: [system_prompt(2000 tokens)] + "读取这个文件"(8 tokens)
请求 3: [system_prompt(2000 tokens)] + "发一封邮件"(7 tokens)
...
```

每次都重新 Prefill 那 2000 个 token 的 system prompt，纯属浪费。

### vLLM 的 Automatic Prefix Caching (APC)

vLLM 把 KV Cache 按 block 管理（参考第 5 章 **PagedAttention**——vLLM 的分页 KV Cache 管理算法，类比操作系统的分页内存管理）。**APC**（Automatic Prefix Caching，自动前缀缓存）的做法：对 token 序列的每个 block 算一个 hash，如果新请求的前缀 hash 匹配已有 cache，直接复用。

```bash
# 启用 APC
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B \
    --enable-prefix-caching
```

效果取决于 prefix 长度和请求量（**QPS**：Queries Per Second，每秒请求数，第 1 章已介绍）：

| System Prompt 长度 | 每请求节省的 Prefill | 100 QPS 下节省的 GPU 算力 |
|-------------------|--------------------|-----------------------|
| 500 tokens | ~25 ms | ~2.5 秒/秒的 GPU 时间 |
| 2000 tokens | ~100 ms | ~10 秒/秒的 GPU 时间 |
| 5000 tokens | ~250 ms | ~25 秒/秒的 GPU 时间 |

Agent 场景下 system prompt 往往包含大量 **tool description**（工具描述，告诉模型每个可用工具的名字、参数、用法的 JSON Schema 文本），轻松超过 2000 tokens。APC 可以把 **TTFT**（Time To First Token，首 token 延迟，第 4 章已介绍）从 200ms 降到 20ms（只需要 Prefill 用户的短 query）。

Anthropic 和 OpenAI 的 API 也提供了 **Prompt Caching**（提示词缓存，云厂商在 API 层提供的前缀复用机制）功能，原理类似。Anthropic 的 Prompt Caching 对缓存命中的 input token 打 9 折（只收 10% 的价格），这对大量调用同一 system prompt 的 Agent 来说省很多钱。

代码示例见 `examples/ch08-inference-optimization/03_prefix_caching_demo.py`。

## 8.5 结构化输出的约束解码

### Agent 为什么需要结构化输出

Agent 调用 **tool**（工具，Agent 框架中模型可以主动调用的外部能力，如搜索、读文件、发邮件等）时需要生成 JSON：

```json
{"tool": "search_web", "arguments": {"query": "vLLM latest version"}}
```

靠 prompt 引导（"请输出 JSON 格式"）不够可靠——模型可能加 markdown 代码块、多输出一段解释、或者 JSON 格式不合法。对 Agent 来说，一个非法 JSON 就意味着 tool 调用失败，需要重试，浪费 token 和时间。

### 约束解码的原理

**约束解码**（Constrained Decoding / Guided Decoding，引导解码）：在每一步 token 生成时，根据目标格式（**JSON Schema**，描述 JSON 数据结构的标准规范 / 正则表达式），屏蔽掉不合法的 token：

```
当前已生成: {"name": "Al
目标 schema: {"name": string, "age": integer}

此时合法的下一步:
  ✓ 任意字符 (继续字符串)
  ✓ " (结束字符串)
  ✗ } (字符串未结束)
  ✗ , (字符串未结束)
  ✗ 数字 (在字符串内)

实现: 把不合法 token 的 logit 设为 -∞ → softmax 后概率为 0
```

这里 **logit** 指模型最后一层线性投影输出、还没经过 softmax 的原始分数（每个 token 一个数，越大代表越想生成）。这个过程用一个**有限状态机**（**FSM**，Finite State Machine，由有限个状态和转移规则构成的计算模型，前端做表单流程时经常画的状态图就是 FSM）驱动。JSON Schema 先被编译成正则表达式，正则再编译成 FSM。每生成一个 token，FSM 前进一步，输出当前状态下合法的 token 集合。

### 性能开销

约束解码的开销主要在两处：
1. **FSM 编译**：把 JSON Schema 编译成 FSM，一次性开销，通常 < 100ms
2. **每步 token masking**：查 FSM 获取合法 token，设置 logit mask。开销很小，< 0.1ms/step

总体对推理速度的影响通常 < 5%，但换来 100% 合法的输出。

### 工具和框架

**outlines**（开源约束解码库，由 dottxt-ai 维护）— 最早的约束解码库，支持 JSON Schema、正则、选择约束：

代码里用到的 **Pydantic**（Python 端基于类型注解定义/校验数据结构的库，类似 TS 的 Zod）：

```python
from outlines import models, generate
from pydantic import BaseModel

class ToolCall(BaseModel):
    tool: str
    arguments: dict

model = models.transformers("Qwen/Qwen2-7B")
generator = generate.json(model, ToolCall)
result = generator("Call a tool to search for: latest vLLM version")
# result 一定是合法的 ToolCall 对象
```

**vLLM 内置 Guided Decoding**（vLLM 自带的引导解码功能） — 直接在 API 参数中指定：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="na")

response = client.chat.completions.create(
    model="Qwen/Qwen2-7B",
    messages=[{"role": "user", "content": "帮我搜索天气"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "tool_call",
            "schema": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"}
                },
                "required": ["tool", "arguments"]
            }
        }
    }
)
```

**SGLang 的约束解码**（SGLang 是另一款高性能推理引擎，靠 RadixAttention 做前缀树式 KV 复用，第 1、6 章已介绍） — 性能最好。SGLang 优化了 FSM 的编译和执行：**batch**（批，把多个请求拼到一起一次性推理）内共享 FSM 状态，减少重复计算。在需要大量 JSON 输出的 Agent 场景中，SGLang 的 constrained decoding 比 vLLM 快 2-3x。

### 选型建议

| 方案 | 适用场景 | 特点 |
|------|---------|------|
| vLLM guided decoding | 生产部署，通过 API 使用 | 开箱即用，性能不错 |
| SGLang | 高并发 Agent 场景 | 约束解码性能最好 |
| outlines | 研究/原型/自定义模型 | 灵活，支持本地模型 |
| 云 API (OpenAI/Anthropic) | 直接用云服务 | 最简单，JSON mode 即可 |

对 Agent 工程师来说，如果你用云 API，直接用 JSON mode 就行。如果自部署，vLLM 或 SGLang 都内置了约束解码，比自己在应用层 parse + retry 可靠得多。

代码示例见 `examples/ch08-inference-optimization/04_constrained_decoding.py`。

---

## 本章小结

### 推理性能指标速查

讨论加速效果前先把指标统一下，后面章节也会反复用到：

- **TTFT（Time to First Token，首 token 延迟，第 4 章已介绍）**：从客户端发请求到收到第一个 token 的时间，等于 prefill 耗时 + 调度排队。决定用户感知的"响应速度"
- **TPOT（Time Per Output Token，每输出 token 时间）**：稳态 decode 阶段每个输出 token 的生成时间，决定流式输出"跟手不跟手"
- **TPS（Tokens Per Second，每秒 token 数，第 4 章已介绍）**：服务端整体每秒吐出的 token 数（所有并发请求加起来），决定服务器扛多少并发、单位 token 成本多少

三个指标常常此消彼长：调大 batch 提升 TPS，但 TPOT 会上升；开 **chunked prefill**（分块预填充，把超长 prompt 的 prefill 切成小块，穿插在 decode 步骤之间执行，避免单条长请求把后续请求堵死）改善长请求的 TTFT，但单请求峰值 TPS 略降。优化时先想清楚业务最敏感的是哪一个。

### 五个核心技术的定位

| 技术 | 解决的问题 | 是否需要手动配置 | 加速效果 |
|------|-----------|----------------|---------|
| FlashAttention | Attention 显存和速度 | 不需要（PyTorch 自动） | 2-4x |
| Speculative Decoding | Decode 阶段 GPU 利用率低 | 需要选 draft model | 1.5-3x |
| KV Cache 压缩 | KV Cache 显存占用 | 简单配置 | 显存减半 |
| Prefix Caching | 重复 prefix 的计算浪费 | 一行配置开启 | TTFT 降 80%+ |
| 约束解码 | 输出格式不合法 | API 参数指定 | 无加速，但避免重试 |

对 Agent 工程师来说，最应该关注的是 **Prefix Caching** 和 **约束解码**——它们直接影响 Agent 的成本和可靠性。FlashAttention 已经是默认开启的，享受就好。Speculative Decoding 在 TTFT 不敏感但 TPS 重要的场景（长文本生成）价值最大。

### 其他值得知道的优化

本章主线之外的两个工程优化，目前都已经默认集成进 vLLM，不需要单独开关，但出问题时排查日志会看到名字：

- **CUDA Graph**（NVIDIA 提供的一种机制，把一连串 GPU **kernel**——也就是 GPU 上执行的并行计算函数——的调用录制成静态图，之后整图一次性提交）：decode 阶段每步的 kernel 调用序列是高度重复的，把这个序列录制成一张"图"以后直接重放，可以省掉每步 0.5-1ms 的 CPU→GPU 调度开销。对短输出场景（每个请求只生成几十个 token）效果尤其明显。vLLM 默认开启，遇到动态 **shape**（张量的维度形状）的边角情况会自动回退到 **eager**（PyTorch 默认的逐行解释执行模式，与图模式相对）
- **FlashInfer**：[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 是专门为 LLM decode 阶段优化的 **attention kernel**（实现 Attention 的底层 GPU 函数）库，相比 FlashAttention 在 **paged KV**（参考 PagedAttention 的分页 KV Cache）+ 长上下文 + 大 batch 的 decode 场景下更快，对 **GQA**（Grouped-Query Attention，分组查询注意力，多个 Q 头共享一组 K/V 头来省 KV Cache 显存，第 2 章已介绍）支持得更好。vLLM v0.6+ 已经把它作为默认后端

另外一个 Agent 工程师常碰到的话题是 **LoRA serving**（**LoRA**：Low-Rank Adaptation，低秩适配，一种只训练几个小矩阵就能微调大模型的技术；LoRA serving 指在线服务时多个 LoRA 适配器共享同一 base 模型）——同一个 base model 配合多个 LoRA 适配器在线动态切换。vLLM 通过 `--enable-lora --max-loras N` 支持这种用法，多个 **adapter**（适配器，LoRA 训练出来的小权重补丁）共享同一份 base 权重，每个请求按 `model` 字段路由到对应适配器。具体训练侧的工作流见第 9 章。

本章没单独展开但要混个脸熟的几个名字（后续章节会用到）：**Continuous Batching**（连续批处理，每生成一个 token 就重新组 batch，让长短请求高效共存，第 1 章已介绍）、**Tensor Parallel**（**TP**，张量并行，把单层权重切分到多卡上协同计算，第 1 章已介绍，第 11 章详细讲）、**Pipeline Parallel**（**PP**，流水线并行，把模型按层切分到多卡，第 11 章详讲）、**Data Parallel**（**DP**，数据并行，每张卡放完整模型、数据切分到不同卡）、**Expert Parallel**（**EP**，专家并行，**MoE**——Mixture of Experts，混合专家模型——的专家网络切分到不同卡）、**Tree Attention**（树形注意力，配合 Medusa/EAGLE 这类多分支推测使用的 attention 变体）。这些是分布式/并行/大模型架构层面的技术，第 11、12 章会专门讲。

还有几个常见的底层库名字，写 CUDA kernel 时绕不开：**Triton**（OpenAI 出的 Python 写 GPU kernel 的 DSL，写起来像 NumPy 但能生成接近 CUDA 性能的 GPU 代码，注意和 NVIDIA Triton Inference Server 同名不同物）、**CUTLASS**（NVIDIA 官方的 C++ 模板库，用来高性能实现 **GEMM**——General Matrix Multiplication，通用矩阵乘法，第 2 章已介绍——和卷积）、**cuBLAS**（NVIDIA 闭源的基础线性代数库，PyTorch 的矩阵乘默认走它）、**xFormers**（Meta 开源的优化 transformer 算子集合，早期 FlashAttention 普及前的事实标准）、**FasterTransformer**（NVIDIA 早期开源的 transformer 推理加速库，已被 **TensorRT-LLM**——NVIDIA 当前主推的 LLM 推理引擎，第 1、6 章已介绍——取代）。这些库的源码风格、组合方式见第 11 章和附录。

**延伸阅读：**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原论文
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135) — Tri Dao et al.
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)
- [Speculative Decoding 论文](https://arxiv.org/abs/2211.17192) — Leviathan et al.
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180) — Kwon et al.（vLLM）
- [StreamingLLM 论文](https://arxiv.org/abs/2309.17453) — Xiao et al.
- [H2O 论文](https://arxiv.org/abs/2306.14048) — Heavy-Hitter Oracle KV 淘汰策略
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — 专为 LLM decode 优化的 attention kernel 库
- [outlines GitHub](https://github.com/dottxt-ai/outlines)
- [vLLM Guided Decoding 文档](https://docs.vllm.ai/en/latest/features/structured_outputs.html)


---

> 本章来自《LLM Infra 从入门到实践》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-llm-infra](https://github.com/diguike/book-llm-infra)
