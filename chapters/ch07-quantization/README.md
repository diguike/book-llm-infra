# 第 7 章 量化：用更少的显存跑更大的模型

量化是目前大模型落地最实用的技术之一。一个 7B 模型用 FP16 跑需要 14GB 显存，量化到 INT4 只要 4GB 左右——这意味着你可以在一张消费级显卡上跑原本需要 A100 的模型。本章会讲清楚量化的原理、主流方案的差异，以及如何动手量化一个真实模型。

> **7.4 节里执行 GPTQ（GPT Quantization，基于二阶梯度信息逐层量化的 PTQ 算法，7.3 节详解）/ AWQ（Activation-aware Weight Quantization，激活感知的权重量化，7.3 节详解）量化的部分只能在 Linux GPU 服务器跑**——校准（calibration，用一小批数据跑前向、统计权重和激活分布以确定量化参数）过程要在 GPU 上做几小时的 forward。Mac 本地可以跳到 7.4 节中的 GGUF（GPT-Generated Unified Format，llama.cpp 生态的单文件量化模型格式，ch04 介绍过）部分：llama.cpp（C/C++ 实现的轻量 LLM 推理引擎，能在 CPU、Metal、CUDA 上跑量化模型）的转换和量化工具在 Apple Silicon（苹果 M 系列 ARM 芯片，集成 CPU/GPU/神经网络单元）上完全跑得动，CPU/Metal（苹果自家的 GPU 图形/计算 API，类似 macOS 上的 CUDA）都可以。

## 7.1 为什么量化有效

### 浮点数精度的基本概念

先搞清楚一个基础问题：模型参数到底占多少空间？

一个模型的参数就是一堆浮点数。不同的数值表示格式，精度和存储空间差异巨大：

| 格式 | 位数 | 每个参数占用 | 数值范围 | 典型用途 |
|------|------|-------------|---------|---------|
| FP32 | 32 bit | 4 bytes | ±3.4×10³⁸ | 传统训练 |
| FP16 | 16 bit | 2 bytes | ±6.5×10⁴ | 混合精度训练/推理 |
| BF16 | 16 bit | 2 bytes | ±3.4×10³⁸ | 大模型训练首选 |
| INT8 | 8 bit | 1 byte | -128 ~ 127 | 量化推理 |
| INT4 | 4 bit | 0.5 bytes | -8 ~ 7 | 激进量化推理 |

**BF16（Brain Floating Point 16-bit，Google Brain 提出的 16 位浮点格式，ch03 已介绍）vs FP16（IEEE 754 半精度浮点）的区别**：FP16 有 10 位尾数、5 位指数，精度高但容易溢出（范围只到 6.5 万）。BF16 有 8 位指数、7 位尾数，范围和 FP32（IEEE 754 单精度浮点，32 位）一样大，但精度低一些。大模型训练普遍用 BF16，因为训练过程中梯度的动态范围很大，溢出比精度损失更致命。

### 一个 7B 模型在不同精度下的显存占用

拿 Qwen2.5-7B（阿里通义千问开源的 70 亿参数大模型，2024 年 9 月发布）举例，它有大约 72 亿个参数：

| 精度 | 参数存储 | 实际推理显存（含 KV Cache（Key-Value Cache，自回归推理时缓存历史 token 的注意力 K/V 张量，ch05 已介绍）等开销） | 推理速度（tokens/s, A100（NVIDIA Ampere 架构旗舰数据中心 GPU，40GB/80GB 显存）） |
|------|---------|--------------------------------|--------------------------|
| FP32 | 28 GB | ~32 GB | 基线 |
| FP16/BF16 | 14 GB | ~18 GB | ~40 tokens/s |
| INT8 | 7 GB | ~10 GB | ~55 tokens/s |
| INT4 (GPTQ) | 3.5 GB | ~6 GB | ~70 tokens/s |
| Q4_K_M (GGUF) | 4.08 GB | ~6.5 GB | ~65 tokens/s（CPU 上也可跑） |

推理显存比纯参数存储多出来的部分主要是 KV Cache、激活值和框架开销。

### 量化的核心 tradeoff

量化本质上是用离散的低精度值来近似连续的高精度值。想象一下，你有一个 0 到 1 之间的浮点数，FP16 可以用 1024 个不同的值来表示它，INT4 只有 16 个值。信息必然会丢失。

但实际效果比你预期的好得多。这背后有几个原因：

1. **模型参数的分布是有规律的**：大部分权重集中在 0 附近，极端值很少。量化方法可以针对这个分布做优化。
2. **模型本身有冗余**：7B 参数里不是每个都关键，很多参数的微小变化对输出影响很小。
3. **分组量化（Group Quantization，又叫 per-group 量化，介于 per-tensor（整张量共享一组参数）和 per-channel（按通道一组参数）之间的折中粒度）**：不是对整个矩阵用一组量化参数，而是每 128 个元素用一组 scale 和 zero-point，大幅减少量化误差。

这里出现的两个术语后面会反复用到，先约定下：**scale**（缩放因子）是浮点值和整数值之间的比例，量化时拿原始浮点除以 scale 再四舍五入；**zero-point**（零点偏移）是整数 0 对应的浮点值，用来处理非对称分布（比如所有权重都是正数）。INT4 对称量化时 zero-point 可以省去，只留一个 scale。

实测数据：Qwen2.5-7B 用 INT4 (AWQ) 量化后，在常见 benchmark（基准测试，类似前端跑 Lighthouse 打分，量化领域常用 MMLU、PPL、GSM8K 等）上的性能下降通常在 1-2% 以内，但显存占用减少了 75%。这个 tradeoff 对绝大多数应用场景来说都是划算的。

## 7.2 PTQ vs QAT

量化方法从大的技术路线上分两类：

### Post-Training Quantization (PTQ)

PTQ（Post-Training Quantization，训练后量化，ch01 已介绍——不需要回到训练阶段，加载已训练好的模型直接量化）在模型训练完之后做量化。你不需要原始训练数据，也不需要重新训练，只需要一小批校准数据（通常几百条文本就够了）来统计权重的分布。

**工作流程**：
1. 加载训练好的 FP16 模型
2. 用校准数据做一次前向传播，收集各层权重和激活值的统计信息
3. 根据统计信息计算量化参数（scale, zero-point）
4. 将权重转换为低精度格式
5. 保存量化后的模型

**优点**：
- 简单快速，几十分钟到几小时就能完成
- 不需要训练数据和 GPU 集群
- GPTQ、AWQ、GGUF 都属于这类

**缺点**：
- 极端量化（如 INT2）下精度损失明显
- 对某些模型结构可能效果不理想

### Quantization-Aware Training (QAT)

QAT（Quantization-Aware Training，量化感知训练——训练时在前向里插入"假量化"节点，让模型自己学着适应量化误差）在训练过程中模拟量化的效果。模型在训练时就"知道"自己最终会被量化，所以会学习对量化更鲁棒的权重。

**工作流程**：
1. 在模型的前向传播中插入 fake quantization（伪量化，前向时做"量化→反量化"模拟低精度误差，权重本身还是 FP32）节点
2. 前向传播时模拟量化效果（加入量化噪声）
3. 反向传播时使用 Straight-Through Estimator（STE，直通估计器——量化函数本身不可导，反传时直接把梯度"直通"原样传过去）绕过量化操作的不可导问题
4. 经过若干 epoch（一轮完整遍历训练集）的训练，权重自然适应了量化

**优点**：
- 精度损失最小，尤其在低比特（INT4、INT2）下优势明显
- 适合对精度要求极高的场景

**缺点**：
- 需要完整的训练流程，计算成本高
- 需要训练数据
- 实践中用得不多，因为 PTQ 的效果已经足够好

**结论**：在 2025 年的实践中，PTQ 是绝对主流。除非你在做 INT2 级别的极端量化，或者你的模型对精度有极端要求，否则直接用 PTQ 就够了。

## 7.3 主流量化方法详解

### GPTQ：基于 Hessian 的逐层量化

GPTQ（Generative Pre-trained Transformer Quantization，[arxiv:2210.17323](https://arxiv.org/abs/2210.17323)，2022 年提出，名字来自论文里把这个方法应用到 GPT 系列模型）是第一个让 INT4 量化大模型真正实用的方法。它的核心思路：

1. **逐层量化**：不是一次性量化整个模型，而是一层一层地处理。量化一层时，用 Hessian 矩阵（二阶导数信息——一阶导是梯度，二阶导描述曲面弯曲程度，用于衡量哪些权重对损失更敏感）来衡量每个权重对输出的影响。
2. **最优量化顺序**：先量化对输出影响最小的权重，然后调整剩余权重来补偿误差。
3. **分组量化**：每 128 个权重共享一组量化参数，平衡精度和压缩比。

**工具链演进**：原始的 AutoGPTQ（HuggingFace 生态最早的 GPTQ 量化库）项目已在 2025 年 4 月归档，不再维护。它的继任者是 [GPTQModel](https://github.com/ModelCloud/GPTQModel)（ModelCloud 团队维护的多算法量化库），不仅支持 GPTQ，还支持 AWQ、GGUF、FP8（详见后文 7.3 节专门小节）等多种量化格式，并且持续更新中。

使用 GPTQModel 量化的基本代码：

```python
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer  # HuggingFace transformers：业界最广泛的预训练模型加载/训练库，类比前端的 React
from datasets import load_dataset        # HuggingFace datasets：配套的数据集加载库

model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 准备校准数据
calibration_dataset = [
    tokenizer(example["text"])
    for example in load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train"
    ).select(range(1024))
]

# 量化配置：4-bit, 每 128 个权重一组
quant_config = QuantizeConfig(bits=4, group_size=128)

# 加载模型并量化
model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration_dataset)
model.save("Qwen2.5-7B-Instruct-GPTQ-Int4")
```

校准数据用 C4 是惯例：C4（Colossal Clean Crawled Corpus，超大规模清洗过的网页爬虫语料）是 Google 训练 T5（Text-to-Text Transfer Transformer，Google 2019 年提出的统一文本到文本的 Transformer 模型）用的清洗过的网页文本，覆盖面广，分布接近通用预训练语料，做通用量化的"对照组"很合适。如果你的模型只在中文/代码这类垂直场景跑，换成对应领域的数据做校准能进一步降低损失。1024 条是社区经验值——再少误差变大，再多边际收益很小。

### AWQ：Activation-Aware Weight Quantization

AWQ（Activation-aware Weight Quantization，激活感知的权重量化，[arxiv:2306.00978](https://arxiv.org/abs/2306.00978)，2023 年提出，MLSys 2024 最佳论文。ch01 提过"激活感知"的术语，这里展开讲）的思路比 GPTQ 更优雅：

**核心洞察**：不是所有权重都一样重要。有些权重通道对应的激活值幅度特别大，这些通道的量化误差会被激活值放大，对输出影响更大。

**做法**：
1. 用校准数据统计每个权重通道对应的激活值幅度
2. 找出"重要通道"（激活幅度大的）
3. 对重要通道的权重做 per-channel scaling（按通道维度逐列缩放，区别于整张量统一 scale 的 per-tensor 做法），在量化前先放大这些权重，让它们在量化时获得更高的有效精度
4. 相应地缩小激活值来保持数学等价

**为什么 AWQ 通常比 GPTQ 好**：
- GPTQ 需要反复调整权重来补偿误差，计算开销大，而且误差可能在层间累积
- AWQ 只做简单的 scaling，计算快且稳定
- AWQ 更好地保护了关键通道，在同等比特下通常精度更高
- AWQ 量化速度也更快，通常是 GPTQ 的 2-3 倍

**工具链**：原始的 AutoAWQ（MIT-Han Lab 维护的 AWQ 量化参考实现）项目在 2025 年 5 月归档。官方推荐使用 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)（vLLM 团队的统一量化工具，支持 AWQ/GPTQ/FP8/INT8 等多种算法）作为替代。不过 GPTQModel 同样支持 AWQ 格式。截至 2026 年，AWQ 已经成为生产环境 INT4 推理的事实标准，vLLM（UC Berkeley 团队开源的高吞吐推理引擎，ch05 主角）、SGLang（LMSYS 团队的另一个高性能推理框架，第 6 章会详细对比）、TensorRT-LLM（NVIDIA 官方推出的基于 TensorRT 的 LLM 推理框架）都内置了优化的 AWQ kernel（在 GPU 上跑的低层算子代码，可以类比成"针对硬件特调的 hot path"）。

### GGUF 量化

GGUF 是 llama.cpp 生态的量化格式，和 GPTQ/AWQ 走的是完全不同的路线。

**定位差异**：GPTQ 和 AWQ 面向 GPU 推理，需要专门的 CUDA kernel（用 CUDA C++ 编写、运行在 NVIDIA GPU 上的算子代码）。GGUF 面向 CPU 和混合推理（CPU + GPU），用的是 llama.cpp 自己的推理引擎。

**量化类型命名规则**：

GGUF 的量化类型看起来很复杂（Q4_K_M、Q5_K_S 之类），其实有规律：

- `Q` + 数字 = 量化比特数（Q4 = 4-bit, Q5 = 5-bit, Q8 = 8-bit）
- `K` = K-quant（llama.cpp 2023 年引入的"非均匀量化"方案，对注意力和 FFN 的关键层用更高比特保护质量）
- `_S` / `_M` / `_L` = Small / Medium / Large，表示变体大小（L 保留更多高精度层）

常见量化类型对比（以 7B 模型为例）：

| 量化类型 | 有效 bpw | 模型大小 | 质量 | 推荐度 |
|---------|---------|---------|------|-------|
| Q2_K | 2.6 | ~2.8 GB | 差，明显降质 | 不推荐 |
| Q3_K_M | 3.3 | ~3.3 GB | 可用，有损失 | 显存极端受限时 |
| Q4_K_M | 4.5 | ~4.1 GB | 好，推荐 | **性价比最高** |
| Q5_K_M | 5.3 | ~4.8 GB | 很好 | **生产环境推荐** |
| Q6_K | 6.6 | ~5.5 GB | 接近 FP16 | 追求质量时 |
| Q8_0 | 8.0 | ~7.0 GB | 几乎无损 | 有充足 RAM 时 |

> bpw = bits per weight，有效每权重比特数（含 scale/zero-point 等开销后的平均位宽）。K-quant 的实际 bpw 不是整数，因为它对不同层用了不同精度。

**GGUF 的适用场景**：
- 本地部署、边缘设备
- 没有 GPU 或 GPU 显存不够的情况
- 开发者自己电脑上快速跑模型做实验
- macOS 上利用 Metal 加速（前文已介绍过 Metal）

### FP8：Hopper 架构上的硬件原生格式

FP8（8-bit Floating Point，8 位浮点，ch03 已介绍过格式定义）不是软件做的近似量化，而是 H100/H200（NVIDIA Hopper 架构的两代旗舰 GPU）和后续 Blackwell（NVIDIA 2024 年发布的下一代架构，代表卡是 B200/B100）架构在硬件 Tensor Core（NVIDIA GPU 内部专门做矩阵乘加的单元，ch03 已介绍）一级直接支持的 8-bit 浮点格式。两种常见变体：E4M3（4 位指数 + 3 位尾数，精度优先）和 E5M2（5 位指数 + 2 位尾数，范围优先），主流推理引擎多用 E4M3 存权重。

相比 INT4：

- **精度**：接近 BF16，主流 benchmark 上掉点通常 < 0.5%，比 INT4 量化损失小一个数量级
- **速度**：H100 上 FP8 GEMM（General Matrix Multiply，通用矩阵乘法，深度学习里大部分计算都归结到这个原语）吞吐量是 BF16 的 2 倍
- **显存**：每参数 1 byte，比 BF16 减半，但只有 INT4 的一半压缩比
- **限制**：必须 Hopper 及以上（H100/H200/H20/B200），A100 不支持

工具链上，[vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) 是 HF（HuggingFace 的常见简称）/vLLM 阵营的官方量化工具，支持把权重、激活、KV Cache 都量到 FP8。vLLM 启动参数里直接加 `--quantization fp8` 加载已经量化好的 FP8 模型，或者把 KV Cache 单独量化：

```bash
vllm serve neuralmagic/Qwen2-7B-Instruct-FP8 \
    --quantization fp8 \
    --kv-cache-dtype fp8
```

如果你有 H100 集群且追求精度/速度平衡，FP8 是目前的首选。A100 还在用，那就老老实实走 AWQ INT4。

### 其他常见工具

除了 GPTQ / AWQ / GGUF，下面这两个名字在量化相关的文档里也经常出现，简单交代下定位避免读者懵：

- **bitsandbytes**（Tim Dettmers 开源的 8-bit / 4-bit 量化库，QLoRA 论文的官方实现配套库）：HuggingFace `transformers` 原生支持的"开箱即用"量化方案，`AutoModel.from_pretrained(..., load_in_4bit=True)` 一行启用，量化是**加载模型时即时做的**（NF4（NormalFloat 4-bit，QLoRA 论文设计的、按权重正态分布优化分桶的 4 位格式）/ FP4（普通的 4-bit 浮点，1 位符号 + 2/1 位指数尾数）），用来快速做实验、跑 PEFT（Parameter-Efficient Fine-Tuning，参数高效微调的总称）/ QLoRA（Quantized LoRA，把基模量化到 4-bit 再叠 LoRA 适配器训练，ch01 已介绍）训练。缺点是不会生成可单独保存的量化权重文件，分发不友好，推理也不如 AWQ kernel 快
- **SmoothQuant**（MIT-Han Lab 2022 年提出的 INT8 量化技术）：针对 INT8 量化中"激活值有少量极端 outlier（离群值，远离主分布的极端大数值，量化时会撑爆量化范围）"的问题，量化前先把激活的尺度按 channel 转移到权重上，让两边都更好量。它已经被吸收进 `llm-compressor`，作为 INT8 路线（W8A8，Weight 8-bit + Activation 8-bit 的简写，表示权重和激活都量到 8-bit）的标准做法。Hopper 之前的卡跑 INT8 推理常用这个

### 方法对比

| 维度 | GPTQ | AWQ | GGUF | FP8 |
|------|------|-----|------|-----|
| 推理设备 | GPU | GPU | CPU / CPU+GPU | GPU（H100+） |
| 典型精度 | INT4, INT8 | INT4 | Q2 ~ Q8 多种 | FP8 (E4M3) |
| 量化速度 | 慢（1-4小时/7B） | 快（20-40分钟/7B） | 快（几分钟） | 快（10-30 分钟/7B） |
| 推理速度（GPU） | 快 | 快 | 较慢 | 极快 |
| 推理速度（CPU） | 不支持 | 不支持 | 好 | 不支持 |
| 生态支持 | vLLM, SGLang, TRT-LLM（TensorRT-LLM 的简称） | vLLM, SGLang, TRT-LLM | llama.cpp, Ollama（在 llama.cpp 之上做了"开箱即用"包装的本地模型运行工具，命令行体验类似 Docker） | vLLM, SGLang, TRT-LLM |
| 精度保持 | 好 | 更好 | 取决于量化类型 | 几乎无损 |
| 推荐场景 | GPU 服务端推理 | GPU 服务端推理（首选） | 本地/边缘/CPU 推理 | H100+ 集群，精度敏感 |

## 7.4 实战：量化一个 7B 模型

下面动手操作。完整代码在 `examples/ch07-quantization/` 目录下。

### 使用 GPTQModel 量化 Qwen2.5-7B

```python
# examples/ch07-quantization/02_gptq_quantize.py
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset

model_id = "Qwen/Qwen2.5-7B-Instruct"
output_dir = "Qwen2.5-7B-Instruct-GPTQ-Int4"

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. 准备校准数据（1024 条，来自 C4 数据集）
calibration_dataset = [
    tokenizer(example["text"])
    for example in load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train"
    ).select(range(1024))
]

# 3. 配置量化参数
quant_config = QuantizeConfig(
    bits=4,           # 4-bit 量化
    group_size=128,   # 每 128 个权重一组
)

# 4. 加载模型
model = GPTQModel.load(model_id, quant_config)

# 5. 执行量化（在 A100 上大约需要 1-2 小时）
model.quantize(calibration_dataset)

# 6. 保存量化模型
model.save(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"量化完成，模型保存到 {output_dir}")
```

### 使用 llama.cpp 转换 GGUF

```bash
# examples/ch07-quantization/04_gguf_convert.sh

# 1. 克隆 llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && make -j

# 2. 从 HuggingFace 下载模型（或用本地路径）
# pip install huggingface-hub
# huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct

# 3. 转换为 GGUF 格式（FP16）
python convert_hf_to_gguf.py ./models/Qwen2.5-7B-Instruct \
    --outfile ./models/qwen2.5-7b-instruct-f16.gguf \
    --outtype f16

# 4. 量化为 Q4_K_M（推荐的平衡选择）
./llama-quantize ./models/qwen2.5-7b-instruct-f16.gguf \
    ./models/qwen2.5-7b-instruct-q4_k_m.gguf Q4_K_M

# 5. 测试推理
./llama-cli -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -p "Hello, how are you?" -n 128
```

### 量化前后效果对比

在 A100 80GB 上的实测数据（Qwen2.5-7B-Instruct）：

| 指标 | FP16 | GPTQ INT4 | AWQ INT4 | Q4_K_M (GGUF) |
|------|------|-----------|----------|---------------|
| 模型大小 | 14.2 GB | 3.9 GB | 3.9 GB | 4.1 GB |
| 推理显存 | 17.8 GB | 6.2 GB | 6.1 GB | N/A (CPU) |
| 生成速度 | 42 tok/s | 68 tok/s | 71 tok/s | 32 tok/s (CPU) |
| 困惑度 (PPL，Perplexity，语言模型在测试集上预测下一个 token 的不确定度，越低越好) | 6.42 | 6.58 | 6.51 | 6.55 |
| MMLU（Massive Multitask Language Understanding，57 个学科的多选题集合，业界最常用的通用能力 benchmark） 分数 | 70.2% | 69.1% | 69.5% | 69.3% |

拆解一下这组数字：
1. **INT4 量化后模型大小缩小 ~3.6 倍**，和理论值（16/4=4 倍）接近，多出来的是量化参数（scale, zero-point）的开销。
2. **GPU 上 INT4 推理反而更快**。因为 LLM 推理是 memory-bound 的（受显存带宽限制而非算力限制，对应概念是 compute-bound）——瓶颈不是计算而是从显存读取权重。权重小了，读取快了，推理就快了。
3. **AWQ 的困惑度略优于 GPTQ**，验证了 AWQ 的 activation-aware 策略确实更有效。
4. **GGUF Q4_K_M 在 CPU 上也能跑到 32 tok/s**，对于本地开发和演示来说完全够用。

> 完整的 benchmark 代码见 `examples/ch07-quantization/05_quantization_benchmark.py`

### 小结

量化是目前 ROI（Return on Investment，投入产出比）最高的模型优化技术。大多数场景下，直接下载 Hugging Face（全球最大的开源模型/数据集托管平台，类比 GitHub 之于代码）上现成的 AWQ/GPTQ 量化模型就够了，不需要自己量化。如果你需要自己量化，优先考虑 AWQ。如果需要在 CPU 或本地跑模型，用 GGUF。

下一章我们聊推理加速技术——FlashAttention（Stanford Tri Dao 等人提出的 IO-aware 注意力算子，把注意力计算重排成对显存带宽友好的形式）、Speculative Decoding（投机解码，用一个小模型先"猜"几个 token、再让大模型一次性验证，大幅减少串行步数）这些让推理更快的黑科技。


---

> 本章来自《LLM Infra 从入门到实践》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)
