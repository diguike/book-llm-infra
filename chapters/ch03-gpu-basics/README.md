# 第 3 章 GPU 与计算基础

## 3.1 GPU vs CPU

### 为什么 LLM 需要 GPU

一句话：LLM 推理的核心操作是矩阵乘法，GPU 做矩阵乘法比 CPU 快 100-1000 倍。

CPU 的设计哲学是"少量核心、高频率、复杂控制逻辑"——适合分支多、逻辑复杂的串行任务。GPU 的设计哲学是"大量核心、较低频率、简单控制逻辑"——适合同一操作并行应用到大量数据。

具体数字（先把下面表头出现的几个术语对上号，其他术语在首次出现时单独解释）：

- **TFLOPS**：Tera FLoating-point Operations Per Second，每秒万亿次浮点运算（10¹² 次/秒），是衡量算力的标准单位。1 TFLOPS ≈ 你在 JS 里跑 `a * b + c` 这种运算每秒一万亿次。
- **FP32 / FP16**：32 位 / 16 位**浮点数**（Floating Point）。位数越低，单次运算更快、占显存更少，但能表示的数值精度也更低。AI 训练和推理大多用 FP16/BF16，FP32 多见于精度敏感的场景。
- **DDR5**：Double Data Rate 5，第 5 代系统内存，装在主板内存插槽上由 CPU 使用，单条带宽 ~40 GB/s。Node 进程里 V8 堆分配的内存就在这里。
- **HBM2e**：High Bandwidth Memory 2e，第 2 代加强版高带宽显存，**和 GPU 芯片封装在一起**（物理距离极短），带宽是 DDR5 的几十倍。GPU 上的显存就是 HBM 系列。

| 指标 | Intel Xeon 8380 (CPU) | NVIDIA A100 (GPU) | 倍数 |
|------|----------------------|-------------------|------|
| 核心数 | 40 | 6912 CUDA + 432 Tensor | ~170x |
| FP32 吞吐 | 2.3 TFLOPS | 19.5 TFLOPS | ~8x |
| FP16 吞吐 | 4.6 TFLOPS | 312 TFLOPS | ~68x |
| FP16 Tensor Core | - | 312 TFLOPS | - |
| 内存带宽 | 204.8 GB/s (DDR5) | 2039 GB/s (HBM2e) | ~10x |

记住一个直觉：**GPU 不只是算得快，更重要的是显存带宽大**。表里 GPU 的内存带宽是 CPU 的 10 倍，这才是 LLM 推理性能的关键。

Llama 2 7B 生成一个 token 需要约 140 亿次浮点运算（14 GFLOPS，G = Giga = 10⁹）。CPU 用 FP16 需要约 3ms（理论最优，实际更慢），A100 用 Tensor Core 只需约 0.045ms。batch size 越大，差距越明显。

### CUDA Core vs Tensor Core

**CUDA Core**：通用浮点运算单元，一个 CUDA Core 每个时钟周期做 1 次 FP32 乘加运算（**FMA**，Fused Multiply-Add，把 `a × b + c` 合成一条硬件指令）。A100 有 6912 个 CUDA Core。

**Tensor Core**：专门为矩阵乘法设计的硬件单元。一个 Tensor Core 一个时钟周期可以做一个 4x4 的矩阵乘加（也就是 128 次 FMA），效率远超 CUDA Core。A100 有 432 个 Tensor Core。

```
CUDA Core（1个周期做 1 次运算）:
  a × b + c → result

Tensor Core（1个周期做 4×4 矩阵乘加）:
  D = A × B + C
  其中 A, B, C, D 都是 4×4 矩阵
  = 64 次乘法 + 64 次加法 = 128 次 FMA
```

Transformer 的核心操作（Q@K^T、Attention@V、FFN 的矩阵乘法）天然适合 Tensor Core。实际上，现代推理引擎（vLLM、TensorRT-LLM）都会尽量把计算放到 Tensor Core 上。

A100 的 Tensor Core 支持的精度：FP16、BF16、TF32、INT8、INT4。H100 新增了 FP8 支持，B200 新增了 FP4——精度越低，吞吐越高。

这里出现了两个 AI 领域特有的浮点格式，提前澄清一下：

- **FP16**（半精度浮点）：16 位浮点，精度高但数值范围窄（最大约 65504），训练时容易出现溢出和下溢。
- **BF16**（Brain Float 16）：同样是 16 位浮点，但重新分配了指数位，**数值范围和 FP32 一样**，代价是精度比 FP16 略低。现代 LLM 训练和推理默认用 BF16，主要就是因为它不容易溢出。
- **TF32**（Tensor Float 32）：NVIDIA 在 A100 引入的内部格式，实际只用 19 位（指数 8 位、尾数 10 位），对用户透明——开启后，FP32 的矩阵乘法会自动降到 TF32 在 Tensor Core 上跑，速度快很多，精度仍接近 FP32。

下面表格里还会出现更低精度的格式：

- **FP8 / FP4**：8 位 / 4 位浮点（Floating Point）。H100 首次支持 FP8，B200 进一步支持 FP4，在更低位宽下保留浮点的动态范围，是新一代推理/训练的主力格式。
- **INT8 / INT4**：8 位 / 4 位整数（Integer）。主要用于量化推理——把原本 FP16 的权重压缩成整数存储，推理时走整数 Tensor Core 路径，第 7 章详细讲。

| 精度 | A100 TFLOPS | H100 TFLOPS | B200 TFLOPS |
|------|-------------|-------------|-------------|
| FP32 | 19.5 | 67 | 90 |
| FP16/BF16 | 312 | 990 | 2250 |
| FP8 | - | 1979 | 4500 |
| INT8 | 624 | 1979 | 4500 |
| FP4 | - | - | 9000 |

INT8 吞吐是 FP16 的 2 倍——这是量化（Quantization）加速的硬件基础。量化的本质是把原本 FP16 的模型权重压缩成 INT8 或 INT4 存储，推理时走整数 Tensor Core 路径计算，显存占用、显存带宽压力同步下降。第 7 章会详细讲量化的实现方式与代价。

## 3.2 显存（VRAM）

### 模型参数占多少显存

公式很简单：**显存 = 参数量 x 每个参数的字节数**

| 精度 | 每参数字节 | 7B 模型显存 | 13B 模型显存 | 70B 模型显存 |
|------|-----------|------------|-------------|-------------|
| FP32 | 4 bytes | 28 GB | 52 GB | 280 GB |
| FP16/BF16 | 2 bytes | 14 GB | 26 GB | 140 GB |
| INT8 | 1 byte | 7 GB | 13 GB | 70 GB |
| INT4 | 0.5 bytes | 3.5 GB | 6.5 GB | 35 GB |

这只是**模型权重**的显存。推理时还有三块额外开销，其中最重要的是 **KV Cache**（Key-Value Cache，键值缓存）—— Transformer 推理时把每一步 Attention 算过的 Key/Value 向量缓存下来，避免重复计算，代价是占用大量显存且随生成长度线性增长。第 4 章会从推理流程角度展开。

### 推理时的显存组成

```
┌────────────────────────────────────────────┐
│              总显存占用                      │
├────────────────────────────────────────────┤
│                                            │
│  模型权重        (固定)                     │
│  ████████████████████  14 GB (7B FP16)     │
│                                            │
│  KV Cache         (随 seq_len × batch 增长) │
│  █████████████         8 GB (示例)          │
│                                            │
│  激活值/中间结果  (临时，前向传播时占用)       │
│  ████                  2 GB (示例)          │
│                                            │
│  CUDA Context + 碎片                        │
│  ██                    1-2 GB               │
│                                            │
├────────────────────────────────────────────┤
│  总计:                 ~25 GB               │
└────────────────────────────────────────────┘
```

以 Llama 2 7B FP16 在 A100 80GB 上为例：
- 模型权重：14 GB（固定）
- KV Cache：取决于 batch size 和 seq_len，可能 1-50 GB
- 激活值：约 2 GB（取决于 batch size）
- CUDA overhead：1-2 GB
- 剩余显存可用于更大的 batch 或更长的 context

### 显存不够时的表现

1. **OOM（Out of Memory）**：最常见。CUDA 直接报错 `RuntimeError: CUDA out of memory`。通常发生在请求突然变长或并发变大时。
2. **无法加载模型**：模型权重都放不下。解决：用量化（INT8/INT4）或切到更大卡。
3. **吞吐下降**：显存紧张时 batch size 被迫缩小，GPU 利用率下降。

解决方案（按优先级）：

| 方案 | 效果 | 代价 |
|------|------|------|
| 量化（INT8/INT4） | 显存减半到 1/4 | 精度略有下降 |
| GQA 模型 | KV Cache 缩小 4x | 需要模型支持 |
| Tensor Parallel | 多卡分摊 | 需要多卡 + NVLink |
| PagedAttention (vLLM) | 减少 KV Cache 碎片 | 工程实现复杂 |
| 换更大显存的 GPU | 从根本解决 | 贵 |

表里几个术语先简单铺垫，后面章节会展开：

- **GQA**（Grouped-Query Attention，分组查询注意力）：让多个 Query head 共享同一组 Key/Value，把 KV Cache 减到 1/4~1/8，第 4、5 章再细讲。
- **Tensor Parallel**（张量并行）：把单个大矩阵切到多张卡上并行计算，需要卡间高速互联，第 11 章展开。
- **NVLink**：NVIDIA 自家的 GPU 卡间高速直连总线（A100 是 600 GB/s，H100 是 900 GB/s），比走主板的 PCIe（~64 GB/s）快十几倍，多卡训练几乎必备。
- **PagedAttention**：vLLM 提出的 KV Cache 分页管理方案，类比操作系统的虚拟内存分页，第 5 章是主角。

## 3.3 计算瓶颈 vs 显存瓶颈

这是理解 LLM 推理性能的关键概念。

### Compute-bound vs Memory-bound

每个计算操作都涉及两步：(1) 从显存读数据到计算单元，(2) 计算。瓶颈在哪一步，决定了优化方向。

用一个简单的模型来思考——**Arithmetic Intensity**（计算强度）：

```
Arithmetic Intensity = FLOPs / Bytes accessed
单位: FLOP/Byte
```

公式里两个变量：**FLOPs**（Floating-point Operations，浮点运算总次数，注意末尾的小写 s 是复数标记，区别于带斜杠的 FLOP/s = 每秒次数）和 **Bytes accessed**（从显存读写的字节数）。两者相除即"每读一个字节能做多少次运算"。

GPU 有一个临界值：**计算吞吐 / 显存带宽**。A100 的临界值：

```
A100: 312 TFLOPS / 2.0 TB/s = 156 FLOP/Byte
```

如果一个操作的 Arithmetic Intensity > 156，它是 compute-bound（计算单元忙不过来）。如果 < 156，它是 memory-bound（计算单元在等数据）。

### Prefill vs Decode

LLM 推理分两个阶段：**Prefill** 是处理用户输入的全部 token，一次并行计算；**Decode** 是后续每一步只生成 1 个新 token，串行执行直到结束。第 4 章会从推理流程的完整视角详细拆解这两个阶段，这里先聚焦它们在 GPU 上的计算特征——也就是为什么同一个模型的 Prefill 和 Decode，性能瓶颈完全不一样。

| 阶段 | 操作 | Arithmetic Intensity | 瓶颈 |
|------|------|---------------------|------|
| Prefill | 大矩阵 × 大矩阵 [n, d] @ [d, d] | ~数百（n 越大越高） | Compute-bound |
| Decode | 向量 × 大矩阵 [1, d] @ [d, d] | ~1（每个权重只用 1 次） | Memory-bound |

**Prefill 阶段：** 输入 n 个 token 一起做矩阵乘法。batch 维度（n）越大，每个权重参数被复用的次数越多，Arithmetic Intensity 越高。n=2048 时，每读一次权重矩阵可以做 2048 次运算——远超 156 的临界值，GPU 的计算单元成为瓶颈。

**Decode 阶段：** 每次只处理 1 个 token（batch=1 时）。[1, 4096] @ [4096, 4096] 的矩阵乘法中，权重矩阵（4096x4096x2 = 32MB）被完整读一遍，但只做了 1x4096x4096 ≈ 33M 次运算。Arithmetic Intensity ≈ 33M / 32M ≈ 1——远低于 156 的临界值，显存带宽成为瓶颈。

这意味着：
- **优化 Prefill → 提升算力**（用更快的 GPU、更高效的 kernel）
- **优化 Decode → 提升显存带宽 或 减少要读取的数据量**（量化、GQA 减少 KV Cache）

这也解释了为什么 batch 很重要：Decode 阶段如果同时处理 32 个请求（batch=32），权重矩阵读一次就能服务 32 个请求，Arithmetic Intensity 提升 32 倍，GPU 利用率大幅提高。这就是 continuous batching 的核心意义。

## 3.4 CUDA 编程的最小认知

你不需要写 CUDA，但需要理解三个概念，才能看懂推理引擎的优化思路。

### Kernel、Thread、Block

CUDA 的执行模型：

```
GPU
 └── Grid（一次 kernel 调用）
      ├── Block 0
      │    ├── Thread 0
      │    ├── Thread 1
      │    ├── ...
      │    └── Thread 255
      ├── Block 1
      │    ├── Thread 0
      │    ├── ...
      │    └── Thread 255
      └── ...
```

**Kernel**：一个在 GPU 上运行的函数。比如"把两个矩阵相加"就是一个 kernel。
**Thread**：最小的执行单元，一个 thread 处理一个（或几个）数据元素。
**Warp**：32 个 thread 组成一个 warp，是 GPU 真正的调度单位——SM 每次选一个 warp 里的 32 个 thread 同步执行同一条指令（**SIMT**，Single Instruction Multiple Threads，单指令多线程；和 CPU 的 SIMD 思路类似，但调度单位是线程而非数据通道）。这就是为什么 CUDA kernel 的 thread 数一般要取 32 的倍数。如果 warp 内的 32 个 thread 走了不同分支（warp divergence），GPU 只能串行执行每条分支，性能会显著下降。
**Block**：一组 thread（通常 128/256/512 个），共享一块高速的 Shared Memory（SRAM）。
**SM**（Streaming Multiprocessor，流式多处理器）：GPU 真正的执行单元，每个 Block 被分配到一个 SM 上执行。A100 有 108 个 SM、H100 有 132 个 SM，每个 SM 可以同时运行多个 Block。下文 Shared Memory 标的 `~192KB per SM` 就是这个 SM 内部专属的高速缓存。
**Grid**：所有 block 的集合。

一个实际例子——向量加法：

```c
// CUDA Kernel: C = A + B
// __global__ 表示这个函数在 GPU 上执行，由 CPU 端发起调用
__global__ void vector_add(float *A, float *B, float *C, int n) {
    // blockIdx.x 是当前 block 的编号，threadIdx.x 是当前 thread 在 block 内的编号
    // 两者组合计算出全局唯一的线程坐标
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];  // 每个 thread 处理 1 个元素
    }
}

// 启动: 256 个 thread/block, 需要多少 block 取决于数据大小
// <<<blocks, threads>>> 是 CUDA 的内核启动语法，指定 grid 和 block 的大小
vector_add<<<(n+255)/256, 256>>>(A, B, C, n);
```

### GPU 内存层次

```
┌─────────────────────────────────────────────┐
│ HBM (显存, 80GB, ~2 TB/s)                    │ ← 模型权重、KV Cache 在这里
│                                             │
│   ┌─────────────────────────────────────┐   │
│   │ L2 Cache (40MB, ~5 TB/s)            │   │
│   │                                     │   │
│   │   ┌─────────────────────────────┐   │   │
│   │   │ Shared Memory / L1 Cache    │   │   │ ← 每个 Block 有 ~192KB
│   │   │ (~192KB per SM, ~19 TB/s)   │   │   │    FlashAttention 利用的就是这层
│   │   │                             │   │   │
│   │   │   ┌─────────────────────┐   │   │   │
│   │   │   │ Registers           │   │   │   │ ← 每个 Thread 私有
│   │   │   │ (最快, ~64KB/SM)    │   │   │   │
│   │   │   └─────────────────────┘   │   │   │
│   │   └─────────────────────────────┘   │   │
│   └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

越上层容量越大但速度越慢，越下层容量越小但速度越快。

### FlashAttention 为什么是 "IO 感知" 的

标准 Attention 的实现：

```python
S = Q @ K.T          # [n, n] 写入 HBM
P = softmax(S)       # [n, n] 读 HBM，写 HBM
O = P @ V            # [n, d] 读 HBM，写 HBM
```

中间结果 S 和 P 是 [n, n] 的大矩阵（n=4K 时 = 64MB），每次都要在 HBM 里读写。这些 HBM 访问是 Attention 计算的主要瓶颈。

FlashAttention 的核心思路：**把 Attention 分成小块（tile），在 Shared Memory（SRAM，Static Random Access Memory，静态随机访问存储——位于 SM 内部，速度比 HBM 快约一个数量级，但容量小得多）里完成计算，避免把中间结果写回 HBM。**

```
标准 Attention:  HBM → 计算 S → 写回 HBM → 读 S → 计算 P → 写回 HBM → 读 P → 计算 O
                 ^^^                ^^^       ^^^              ^^^       ^^^
                 5 次 HBM 读写（n×n 级别）

FlashAttention: HBM → SRAM → 分块计算 S,P,O → 写回 HBM
                ^^^                             ^^^
                只有 2 次 HBM 读写（n×d 级别）
```

FlashAttention 2 在 A100 上把 Attention 的速度提升了 2-4 倍，不是靠减少计算量（FLOPs 一样多），而是靠减少 HBM 读写次数。这就是 "IO-aware" 的含义——算法设计考虑了硬件的内存层次。

## 3.5 GPU 选型

### 主要 GPU 对比

| GPU | 架构 | FP16 TFLOPS | 显存 | 显存带宽 | 互联 | 发布 |
|-----|------|-------------|------|----------|------|------|
| A100 80GB | Ampere | 312 | 80GB HBM2e | 2.0 TB/s | NVLink 600GB/s | 2020 |
| H100 80GB | Hopper | 990 | 80GB HBM3 | 3.35 TB/s | NVLink 900GB/s | 2023 |
| H200 | Hopper | 990 | 141GB HBM3e | 4.8 TB/s | NVLink 900GB/s | 2024 |
| B200 | Blackwell | 2250 | 192GB HBM3e | 8.0 TB/s | NVLink 1800GB/s | 2025 |
| L40S | Ada | 366 | 48GB GDDR6X | 864 GB/s | PCIe 4.0 | 2023 |
| RTX 4090 | Ada | 330 | 24GB GDDR6X | 1008 GB/s | PCIe 4.0 | 2022 |

> 数据来源：NVIDIA 各代架构白皮书与官方数据手册。FP16 为带稀疏关闭的 Tensor Core 吞吐（dense throughput）。B200 数据参考 [NVIDIA Blackwell Architecture Technical Brief](https://resources.nvidia.com/en-us-blackwell-architecture)。

表里还有几个第一次出现的术语：

- **GDDR6X**：Graphics Double Data Rate 6X，专为消费级显卡设计的显存（相比 HBM 容量更小、带宽更低，但成本低得多），出现在 RTX 4090、L40S 这类卡上。
- **PCIe**：Peripheral Component Interconnect Express，主板上 CPU 与 GPU 之间的通用总线。PCIe 4.0 x16 实际带宽约 64 GB/s，比 NVLink 慢一个数量级——多卡训练里如果只能走 PCIe，跨卡通信会成为瓶颈。
- **HBM3 / HBM3e**：HBM2e 的下一代，带宽进一步翻倍，容量也更大（H200 上到 141GB，B200 上到 192GB）。

选型建议：

**个人开发/学习：** RTX 4090（24GB）可以跑 7B FP16 或 13B INT4。性价比最高的学习卡。

**生产推理（小规模）：** L40S 或 A100 40GB。L40S 价格约为 A100 的一半，推理性能接近，但显存带宽低，Decode 速度会差一些。

**生产推理（大规模）：** H100 或 H200。H200 的 141GB 显存意味着单卡就能跑 70B INT4 模型。如果跑长 context（128K+），H200 的大显存优势明显。

**旗舰方案：** B200。FP16 算力是 H100 的 2.3 倍，显存带宽是 2.4 倍。FP4 推理可以在单卡上跑 70B 模型。

### 国内常见 GPU

由于出口管制，A100/H100 在国内受限。常见的替代方案：

| GPU | 来源 | FP16 TFLOPS | 显存 | 生态成熟度 |
|-----|------|-------------|------|-----------|
| A800 | NVIDIA（中国特供） | 312 | 80GB HBM2e | 高（兼容 A100） |
| H800 | NVIDIA（中国特供） | 990 | 80GB HBM3 | 高（兼容 H100，NVLink 降速） |
| 910B | 华为昇腾 | ~320 | 64GB HBM2e | 中（CANN 生态，下方注释） |
| MTT S4000 | 摩尔线程 | ~200 | 48GB | 低 |

A800/H800 和 A100/H100 的计算能力完全一样，只是卡间互联（NVLink）带宽被削减了。对单卡推理场景（7B-13B 模型）影响很小，对多卡并行训练影响较大。

表里 **CANN**（Compute Architecture for Neural Networks）是华为昇腾 NPU 的软件栈，对标 NVIDIA 的 CUDA——做的事差不多（提供算子库、编译器、Runtime），但生态成熟度差距比较大，主流推理引擎（vLLM、SGLang）的昇腾支持仍在追赶。

### 云 GPU 实例参考价格

> 以下价格数据截至 2026 年初，仅供数量级参考，请以云厂商官网最新定价为准。

以按量计费为参考：

| 云厂商 | 实例 | GPU | 参考价格 |
|--------|------|-----|----------|
| 阿里云 | ecs.gn7i-c8g1.2xlarge | 1x A10 24GB | ~15 元/小时 |
| 阿里云 | ecs.gn7-c12g1.3xlarge | 1x A100 80GB | ~35 元/小时 |
| 腾讯云 | GN10Xp.2XLARGE40 | 1x A100 40GB | ~28 元/小时 |
| AutoDL | - | 1x A100 80GB | ~5 元/小时（竞价） |
| AutoDL | - | 1x RTX 4090 24GB | ~2 元/小时（竞价） |

对于学习和开发，AutoDL 等 GPU 云平台的竞价实例性价比最高。生产环境建议用包月实例或阿里云/腾讯云的专业 GPU 实例。

---

**延伸阅读：**

- [NVIDIA A100 Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)
- [NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) - H100 架构白皮书
- [NVIDIA H100 产品页](https://www.nvidia.com/en-us/data-center/h100/) - 官方数据手册和规格表
- [NVIDIA Blackwell Architecture Technical Brief](https://resources.nvidia.com/en-us-blackwell-architecture) - B200/B100 架构与算力数据
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) - Horace He 的经典文章
- [GPU Puzzles](https://github.com/srush/GPU-Puzzles) - 用游戏学 CUDA 概念

## 代码示例

| 示例 | 说明 | 硬件要求 |
|------|------|---------|
| [01_gpu_info.py](../../examples/ch03-gpu-basics/01_gpu_info.py) | 获取 GPU 信息和显存使用 | GPU (any) |
| [02_memory_calc.py](../../examples/ch03-gpu-basics/02_memory_calc.py) | 计算不同模型/精度的显存需求 | CPU |
| [03_cpu_vs_gpu_benchmark.py](../../examples/ch03-gpu-basics/03_cpu_vs_gpu_benchmark.py) | 矩阵运算 CPU vs GPU 对比 | GPU (any) |


---

> 本章来自《LLM Infra 从入门到实践》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-llm-infra](https://github.com/diguike/book-llm-infra)
