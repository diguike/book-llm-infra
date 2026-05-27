# 第 11 章 分布式训练基础

> **进阶章节。** 如果你的目标是推理部署和微调（QLoRA 单卡即可），可以先跳过本章。当你需要从零预训练模型或在多卡上做 Full Fine-tuning（全参数微调，参见第 9 章）时再回来。

> **本章所有代码示例只在 Linux GPU 服务器（多卡环境）执行。** 单机多卡（4-8 张 GPU）即可跑通 DDP（Distributed Data Parallel，分布式数据并行）、ZeRO（Zero Redundancy Optimizer，零冗余优化器）、FSDP（Fully Sharded Data Parallel，完全分片数据并行）示例；Pipeline Parallelism（流水线并行）和真正的 3D parallelism（三维并行，DP+TP+PP 的组合）需要多台机器组成的集群。Mac 本地可以阅读概念部分，但任何代码片段都跑不起来。

前两章我们在单卡上搞定了微调和对齐。但现实中，单卡很快就不够用了：

- 模型太大，一张卡放不下（70B 模型 bf16（bfloat16，16 位脑浮点，常用训练精度）需要 140GB）
- 数据太多，单卡训练太慢
- 要做 Full Fine-tuning 或预训练，显存和算力都不够

这一章我们讲分布式训练的核心概念：数据并行、模型并行、ZeRO、FSDP，以及底层的网络通信。不追求面面俱到，但确保你理解每种方案在解决什么问题、什么时候该用哪个。

## 11.1 数据并行

### 集合通信原语：先认识 3 个名字

集合通信原语（collective communication primitive）是多机多卡之间约定的标准通信操作，类比前端里的 `Promise.all` —— 一组进程协同完成一次数据交换。后面会反复出现的三个通信操作，先用一张表理清：

| 原语 | 输入 → 输出 | 一句话理解 | 典型使用 |
|------|-----------|----------|---------|
| **AllReduce**（全规约） | 各卡一份本地张量 → 各卡都拿到"全局求和/求平均"后的张量 | 大家把数据凑一起算个总账，结果发给所有人 | DDP 同步梯度 |
| **AllGather**（全收集） | 各卡一份分片 → 各卡都拿到"完整拼接"后的张量 | 把各家的拼图拿来，每人得到完整的图 | ZeRO Stage 3 / FSDP 前向时拿回完整参数 |
| **ReduceScatter**（规约-分发） | 各卡一份完整张量 → 各卡拿到求和后的 **一个分片** | 大家凑一起算总账，但每人只领自己负责的那段 | ZeRO Stage 2/3 同步梯度 |

数学关系：`AllReduce = ReduceScatter + AllGather`。这正是下面要讲的 ring all-reduce（环形全规约，一种把通信总量降到与卡数几乎无关的 all-reduce 实现）的实现方式。

### 最简单的分布式策略

数据并行（Data Parallelism, DP：每张卡持有完整模型副本，各处理一部分数据）的思路最直观：

1. 每张卡上放一份完整的模型副本
2. 一个 batch（一次梯度更新使用的样本批次）的数据切成 N 份，每张卡处理一份
3. 每张卡各自做 forward（前向传播，算预测）+ backward（反向传播，算梯度），算出梯度（gradient，损失对参数的偏导，告诉优化器参数该往哪个方向移动）
4. 所有卡的梯度做 all-reduce（求平均）
5. 每张卡用平均梯度更新自己的模型参数

效果等价于在单卡上用 N 倍的 batch size（批大小）训练。

### Ring All-Reduce：直觉解释

朴素的"所有卡都发数据到一张主卡再广播（broadcast，一对多广播原语，把一份数据从根节点发给所有其他节点）"会让主卡成为带宽瓶颈。现代深度学习框架（PyTorch DDP、NCCL，NCCL = NVIDIA Collective Communications Library，下面 11.1 末尾会展开）几乎都用 **ring all-reduce** 算法：N 张卡逻辑上排成一个环，**每张卡只和左右两个相邻的卡通信**。

整个 all-reduce 分成两个阶段，每个阶段都跑 N-1 步：

1. **Reduce-Scatter 阶段**：把每张卡上的梯度张量切成 N 个 chunk（数据块）。每一步，每张卡把自己负责的 chunk 加上邻居发来的对应 chunk，再传给下一个邻居。N-1 步之后，每张卡都拿到了"某一个 chunk 的全局求和结果"。
2. **All-Gather 阶段**：每张卡把自己手里那块"完整 chunk"沿着环传一圈。再过 N-1 步，每张卡都拿到了完整的全局求和结果。

**单卡传输量**：每个阶段，每张卡发送约 $\frac{N-1}{N} \times P$ 数据（P 是梯度总字节数），两个阶段合计 $\frac{2(N-1)}{N} \times P$。注意这个公式和 N 几乎无关——8 卡和 64 卡，每张卡的传输量都接近 $2P$。这就是 ring 算法的精妙之处：通信总量不随 N 线性增长。

### All-Reduce 的通信开销

接着上面的公式算具体例子。假设模型有 $P$ 个参数，每个参数的梯度是 bf16（2 bytes）：

以 7B 模型为例：

- 梯度大小：7B × 2 bytes = 14 GB
- 在 ring all-reduce 中，每张卡需要发送和接收约 $2 \times \frac{N-1}{N} \times 14$ GB 的数据
- 8 卡通过 NVLink（NVIDIA 自家的 GPU 间高速互联，单机内做 GPU-GPU 直连用）（900 GB/s 双向带宽）：通信时间约 **0.05 秒**
- 8 卡通过 PCIe（PCI Express，连 GPU、网卡的通用总线）Gen4（64 GB/s 双向带宽）：通信时间约 **0.8 秒**

如果一个训练 step（一次完整的前向+反向+参数更新）的计算时间是 2 秒，那 NVLink 下通信开销可以忽略（2.5%），但 PCIe 下就有 30%+ 的开销了。这就是为什么训练集群都用 NVLink。

### PyTorch DDP

PyTorch 的 DistributedDataParallel（DDP）是数据并行的标准实现。

> **NCCL 是什么？** NCCL（NVIDIA Collective Communications Library，读作"nickel"）是 NVIDIA 实现的 GPU 集合通信库，专门针对 GPU-GPU 直接传输做了优化——不管底层是 NVLink、PCIe 还是 InfiniBand（IB，下文 11.5 会展开，跨机器的低延迟 RDMA 网络），框架都走它。上面讲的 ring all-reduce、AllGather、ReduceScatter 这些原语，PyTorch 在 GPU 上的实现底层都调用 NCCL。GPU 训练必须用 `nccl` backend（通信后端，PyTorch 里负责实际收发字节的实现）；只有 CPU 训练或者 Mac 本地调试才会用 `gloo`（Facebook 开源的跨平台 CPU 通信库，对应 MPI/Message Passing Interface 这类传统 HPC 通信协议在深度学习里的角色）。文档：[docs.nvidia.com/deeplearning/nccl/user-guide/docs/](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组：GPU 训练用 nccl，CPU/Mac 调试才用 gloo
# local_rank：当前进程在本机内的 GPU 序号（0~N-1），区别于全局 rank（跨机器的进程序号）
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 模型放到对应 GPU
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 数据用 DistributedSampler 切分
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)

# 训练循环和单卡完全一样
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

启动方式：

```bash
torchrun --nproc_per_node=4 train.py
```

DDP 的优点是简单、通信效率高（梯度计算和通信可以 overlap，即 communication-computation overlap，通信-计算重叠：算后面层梯度的同时把前面层的梯度发出去，把通信时间藏进计算里）。缺点是每张卡都要放一份完整的模型，所以显存没有节省——模型太大就放不下。

完整代码见 `examples/ch11-distributed-training/01_ddp_basic.py`。

## 11.2 模型并行

当模型大到一张卡放不下时，就需要模型并行。

### Tensor Parallelism（张量并行）

Tensor Parallelism（TP，张量并行：把单个层内部的大矩阵按行或列切到多张卡上）把一个层的矩阵切分到多张卡上。

以一个线性层 $Y = XW$ 为例，$W \in \mathbb{R}^{d \times d}$：

- 把 $W$ 按列切成 $N$ 份：$W = [W_1, W_2, ..., W_N]$
- 每张卡拿一份 $W_i$，计算 $Y_i = X W_i$
- 最后把 $Y_i$ 拼起来（或 all-reduce）得到完整的 $Y$

优点：
- 每层都并行，延迟低
- 线性扩展显存

缺点：
- 每层都需要通信（all-reduce 或 all-gather），对带宽要求极高
- 通常只在同一台机器的 GPU 之间使用（NVLink 连接）

Tensor Parallelism 在推理时（vLLM 的 `--tensor-parallel-size`）比训练时更常见，因为推理对延迟更敏感。

### Pipeline Parallelism（流水线并行）

Pipeline Parallelism（PP，流水线并行：按层切分模型，每张卡放一段连续的层，像工厂流水线那样让样本依次穿过各段）把模型的不同层放到不同卡上：

- GPU 0：Layer 0-7
- GPU 1：Layer 8-15
- GPU 2：Layer 16-23
- GPU 3：Layer 24-31

问题是，朴素的 PP 会导致严重的"气泡"（bubble，流水线里 GPU 因前后依赖无活可干而空转的那段时间）：GPU 0 算完 forward 后，要等 GPU 1、2、3 依次算完才能开始 backward。大部分时间 GPU 都在空闲。

解决方案是 **Micro-batching**（微批切分，把一次 batch 拆成多个小 micro-batch，让前一段 GPU 处理新样本的同时后一段还在算旧样本）：把一个 batch 切成多个 micro-batch，形成流水线：

```
GPU 0: [F1][F2][F3][F4]          [B4][B3][B2][B1]
GPU 1:     [F1][F2][F3][F4]      [B4][B3][B2][B1]
GPU 2:         [F1][F2][F3][F4]  [B4][B3][B2][B1]
GPU 3:             [F1][F2][F3][F4][B4][B3][B2][B1]
```

这样就把空闲时间（bubble）从 $\frac{N-1}{N}$ 降低到 $\frac{N-1}{N+M-1}$，$M$ 是 micro-batch 的数量。

代入具体数字感受一下：4 张 GPU（N=4），如果不切 micro-batch，bubble 占比是 $3/4 = 75\%$——大部分时间 GPU 都在等。把 batch 切成 M=8 个 micro-batch，bubble 降到 $3/11 \approx 27\%$。继续把 M 调到 32，bubble 降到 $3/35 \approx 9\%$。这就是为什么实际部署 PP 时 micro-batch 数量都开得很大。

优点：
- 通信量小（只在层边界传激活值（activation，每层前向计算产生的中间张量，反向传播时还要用到，所以默认要存下来））
- 适合跨机器使用

缺点：
- 不可避免的 bubble 开销
- 负载均衡难（每层的计算量可能不同）

### 实际选择

大规模训练通常组合使用多种并行策略（称为 3D parallelism）：

- **TP**：同一台机器的 GPU 之间（需要高带宽）
- **PP**：跨机器（带宽要求低一些）
- **DP**：在 TP+PP 组的基础上做数据并行

比如用 64 张 GPU 训练一个大模型：
- 8 张卡为一组做 TP（一台机器内）
- 2 组做 PP（跨 2 台机器）
- 4 路 DP（4 份数据并行副本）
- 总共：8 × 2 × 4 = 64 张卡

3D parallelism 的工业实现参见 NVIDIA 的 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)（NVIDIA 开源的大模型训练框架，3D 并行的工业级参考实现），相关论文 [arxiv 1909.08053](https://arxiv.org/abs/1909.08053)。这套代码是目前几乎所有大模型预训练（包括 GPT-3、Llama、Qwen 等）的实际起点。同类的训练框架还有微软的 DeepSpeed（下节展开）、Meta 的 FairScale（PyTorch 生态里更早期的分片训练库，FSDP 的前身）、潞晨科技的 ColossalAI（国产分布式训练框架，特点是配置较简单）、NVIDIA 的 NeMo（在 Megatron-LM 之上封装的端到端训练/推理框架）。

## 11.3 ZeRO 优化

### DeepSpeed 的核心贡献

微软的 [DeepSpeed](https://github.com/microsoft/DeepSpeed)（微软开源的大模型训练/推理框架）（[文档](https://deepspeed.readthedocs.io/)）提出了 ZeRO（Zero Redundancy Optimizer，零冗余优化器，[arxiv 1910.02054](https://arxiv.org/abs/1910.02054)），核心洞察是：

> 在数据并行中，每张卡都保存完整的模型参数、梯度和优化器状态（optimizer state，优化器为每个参数额外维护的统计量，Adam 这类算法里包含一阶/二阶动量等，通常是参数大小的好几倍），这里有大量冗余。我们可以把这些状态分片（shard，按卡数切成若干份，每张卡只持有一份）到不同卡上。

### ZeRO 三个 Stage

ZeRO 一共有三个分片层级（Stage 1/2/3），分片得越彻底，省的显存越多，需要的通信也越多。三种 Stage 的分片范围对比（X 表示每卡都存完整副本，▮ 表示分片到 1/N）：

```
                 参数 (P)    梯度 (G)    优化器状态 (OS)
─────────────────────────────────────────────────────
Stage 0 (DDP)      X           X            X
Stage 1            X           X            ▮ (1/N)
Stage 2            X           ▮ (1/N)      ▮ (1/N)
Stage 3            ▮ (1/N)     ▮ (1/N)      ▮ (1/N)
```

Stage 越高分片越彻底，显存越省，但每个 step 需要的通信也越多（Stage 3 在 forward 和 backward 前都要 all-gather 参数）。

以 7B 模型、8 卡 DP 为例（每卡单独计算）：

**ZeRO Stage 1：分片优化器状态**

- 模型参数：每张卡 15.2 GB（完整副本）
- 梯度：每张卡 15.2 GB（完整副本）
- 优化器状态：每张卡 60.8 / 8 = **7.6 GB**（只存 1/8）
- 总计：~38 GB / 卡（原来是 ~95 GB）

**ZeRO Stage 2：分片优化器状态 + 梯度**

- 模型参数：每张卡 15.2 GB（完整副本）
- 梯度：每张卡 15.2 / 8 = **1.9 GB**（只存 1/8）
- 优化器状态：每张卡 **7.6 GB**（只存 1/8）
- 总计：~24.7 GB / 卡

**ZeRO Stage 3：分片所有状态**

- 模型参数：每张卡 15.2 / 8 = **1.9 GB**（只存 1/8）
- 梯度：每张卡 **1.9 GB**
- 优化器状态：每张卡 **7.6 GB**
- 总计：~11.4 GB / 卡

显存节省对比（7B，8 卡 DP）：

| Stage | 每卡显存 | 节省比例 |
|-------|---------|---------|
| 无 ZeRO | ~95 GB | - |
| Stage 1 | ~38 GB | 60% |
| Stage 2 | ~24.7 GB | 74% |
| Stage 3 | ~11.4 GB | 88% |

### 什么时候用哪个 Stage

**ZeRO Stage 2**：最常用的选择。

- 通信开销和普通 DDP 接近（梯度的 reduce-scatter + all-gather）
- 显存节省显著
- 计算效率几乎不受影响
- 适用场景：模型能在单卡放下，但 optimizer 放不下

**ZeRO Stage 3**：模型本身都放不下时用。

- 需要额外的 all-gather 来获取完整参数（forward 和 backward 前）
- 通信量是 Stage 2 的约 1.5 倍
- 计算效率有 10-20% 的损失
- 适用场景：大模型 Full FT（Full Fine-tuning，全参数微调），或者显存非常紧张

实际建议：先试 Stage 2，不够再用 Stage 3。

DeepSpeed 还提供 **ZeRO-Offload**（把优化器状态/梯度从 GPU 显存搬到 CPU 内存）和更激进的 **ZeRO-Infinity**（进一步用 NVMe 磁盘做参数仓库），思路都是用更慢的存储换显存空间——前者代价小，后者只在显存极度紧张时才用得上。

### DeepSpeed 配置

一个典型的 ZeRO Stage 2 配置：

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

和 HuggingFace Transformers（Hugging Face 出品的 NLP 模型库，PyTorch/TF 生态的事实标准）集成非常简单——在 TrainingArguments 里指定 deepspeed 配置文件路径即可。

完整配置和代码见 `examples/ch11-distributed-training/03_deepspeed_config.json` 和 `04_deepspeed_train.py`。

### Checkpoint sharding：分片下怎么保存模型

ZeRO Stage 3（以及 FSDP 的 FULL_SHARD 模式）下，每个进程手里只有参数的 1/N。直接 `model.state_dict()`（PyTorch 模型权重字典）拿到的就是残缺的，每张卡都保存一份会得到 N 个互不重叠的碎片——既不能直接加载，也不能直接给推理引擎用。两种主流的保存策略：

- **聚合再保存（full state dict）**：在 rank 0（rank 是分布式训练中的全局进程编号，rank 0 通常是主进程，负责日志、保存等"独苗"工作）上用 all-gather 把所有参数拼回完整模型再写盘。优点是保存出来就是一个标准 checkpoint（训练中保存的模型快照，包含权重和必要的训练状态）文件，可以直接被 vLLM 或 transformers 加载；缺点是聚合时 rank 0 需要装下完整模型，70B 模型这样做会 OOM（Out Of Memory，显存/内存耗尽报错）。DeepSpeed 的对应 API 是 `engine.save_16bit_model()`，FSDP 用 `StateDictType.FULL_STATE_DICT`。
- **分片保存（sharded state dict）**：每张卡保存自己负责的那一片，得到 N 个文件加一份元数据。优点是不需要任何卡装下完整模型，写盘也并行；缺点是这些文件不能直接被推理引擎使用，要先用配套工具（DeepSpeed 的 `zero_to_fp32.py`、PyTorch 的 `dcp_to_torch_save`）合并成单个 checkpoint。FSDP 用 `StateDictType.SHARDED_STATE_DICT`。

实战经验：**训练中途的 checkpoint 用 sharded（快、不会 OOM），最终发布的模型用 full（方便部署）**。

## 11.4 FSDP

### PyTorch 原生的 Fully Sharded Data Parallel

FSDP（Fully Sharded Data Parallel，完全分片数据并行）是 PyTorch 原生的分片数据并行——1.11 引入 beta，2.0 起正式稳定，目前推荐用 PyTorch 2.0+ 的 API（[官方文档](https://pytorch.org/docs/stable/fsdp.html)）。思路和 DeepSpeed ZeRO Stage 3 基本一样：

- 把模型参数、梯度、优化器状态都分片
- forward 时 all-gather 收集完整参数
- backward 后 reduce-scatter 分发梯度
- 不需要的参数及时释放（节省显存）

### FSDP vs DeepSpeed ZeRO

| 维度 | FSDP | DeepSpeed ZeRO |
|-----|------|---------------|
| 生态 | PyTorch 原生 | 独立库 |
| 配置 | Python API | JSON 配置文件 |
| ZeRO Stage | 类似 Stage 3（也支持 Stage 2 行为） | Stage 1/2/3 |
| CPU Offload（CPU 卸载，把不当前用的张量放到 CPU 内存腾出显存） | 支持 | 支持 |
| 混合精度（mixed precision，前向/反向用 bf16/fp16 算，关键状态保留 fp32，节省显存的同时尽量保证收敛） | 原生支持 | 原生支持 |
| HF（HuggingFace 生态）集成 | 通过 Accelerate（HF 出的分布式启动器，统一封装 DDP/FSDP/DeepSpeed） | 原生支持 |
| 社区活跃度 | Meta 主推 | 微软主推 |

实际上两者在大多数场景下性能接近。选择建议：

- 如果你用 HuggingFace 生态，两个都方便用
- 如果你从零搭训练框架，FSDP 更"原生"
- 如果需要 ZeRO Stage 1/2 的灵活性，用 DeepSpeed
- 如果需要 CPU Offload 到极致，DeepSpeed 的 ZeRO-Infinity（在 ZeRO-Offload 基础上进一步把参数搬到 NVMe 磁盘）更成熟

### FSDP 使用方式

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

# 混合精度策略
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# 包装模型
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 类似 ZeRO Stage 3
    mixed_precision=mp_policy,
    device_id=torch.cuda.current_device(),
)
```

`ShardingStrategy` 的选项：

- `FULL_SHARD`：类似 ZeRO Stage 3，全分片
- `SHARD_GRAD_OP`：类似 ZeRO Stage 2，只分片梯度和优化器
- `NO_SHARD`：等同于 DDP，不分片

通过 HuggingFace Accelerate（[FSDP 使用指南](https://huggingface.co/docs/accelerate/usage_guides/fsdp)）使用更简单：

```yaml
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_state_dict_type: SHARDED_STATE_DICT
  mixed_precision: bf16
```

完整代码见 `examples/ch11-distributed-training/02_fsdp_train.py`。

## 11.5 训练集群的网络

分布式训练的瓶颈往往不在 GPU 计算，而在 GPU 之间的通信。

### GPU 间通信技术

**NVLink**：NVIDIA 的高速 GPU 互联。

- NVLink 3.0（A100）：双向 600 GB/s
- NVLink 4.0（H100）：双向 900 GB/s
- 通常连接同一台机器内的 GPU

**NVSwitch**：NVIDIA 的 GPU 间交换芯片，把一台机器内所有 GPU 连成全互联（任意两卡都能直连，不用绕路）。

- A100 DGX（NVIDIA 官方 8 卡 A100 服务器型号）：8 GPU 通过 NVSwitch 全互联，任意两张卡 600 GB/s
- H100 DGX：8 GPU 全互联，任意两张卡 900 GB/s

**PCIe**：老牌通用接口。

- PCIe Gen4 x16：双向 ~32 GB/s
- PCIe Gen5 x16：双向 ~64 GB/s
- 带宽比 NVLink 低一个数量级，不适合 TP

**InfiniBand（IB）**：跨机器的高速网络。底层是 RDMA（Remote Direct Memory Access，远程直接内存访问，绕过 CPU 和操作系统内核，让网卡直接读写远端内存，延迟和 CPU 占用都极低），上层走 IB 协议栈。

- HDR InfiniBand：200 Gbps = 25 GB/s
- NDR InfiniBand：400 Gbps = 50 GB/s
- 延迟极低（~1μs），适合大规模集群
- 配合 **GPUDirect**（NVIDIA 让 GPU 显存直接和网卡/存储交换数据，不绕道 CPU 内存）效果最好

**RoCE（RDMA over Converged Ethernet，把 RDMA 跑在以太网上）**：基于以太网的 RDMA。

- 成本比 InfiniBand 低
- 性能接近，但延迟稍高
- 很多云厂商用这个

### 通信带宽对训练速度的影响

我们算一个具体例子。假设你在做 7B 模型的 DDP 训练，每步需要 all-reduce 14 GB 的梯度：

| 网络 | 带宽 | 通信时间（8 卡） | 占比（2s/step） |
|------|-----|--------------|---------------|
| NVLink 4.0 | 900 GB/s | ~0.03s | 1.5% |
| NVLink 3.0 | 600 GB/s | ~0.05s | 2.5% |
| PCIe Gen5 | 64 GB/s | ~0.4s | 17% |
| InfiniBand NDR | 50 GB/s | ~0.5s | 20% |
| 25G 以太网 | 3.1 GB/s | ~8s | 80% |

可以看到：

- **机器内**：必须用 NVLink/NVSwitch，PCIe 勉强可以做 DP，不适合 TP
- **机器间**：至少用 InfiniBand 或高速 RoCE，普通以太网根本跑不动

### 云上的 GPU 集群网络

如果你在云上租 GPU，网络拓扑直接影响训练效率：

**阿里云**：

- GPU 云服务器（ecs.gn7i）：单机 8 卡 A10，NVLink 互联
- 灵骏智算集群：A100/H100，机器间 RDMA 网络，适合大规模训练
- PAI 平台封装了分布式训练的细节

**腾讯云**：

- GPU 云服务器（GN10Xp）：单机 8 卡 A100，NVLink 互联
- 高性能计算集群（HCC）：InfiniBand 互联，适合多机训练

**AWS**：

- p5 实例：8 卡 H100，NVLink + EFA（Elastic Fabric Adapter，AWS 自研的 RDMA 网卡，相当于云上版的 InfiniBand）网络（400 Gbps）
- p4d 实例：8 卡 A100，NVLink + EFA 网络（400 Gbps）

选择建议：

- 单机 8 卡足够你的任务 → 租一台 8 卡机器就好，不用关心机器间网络
- 需要多机 → 必须选有高速互联（RDMA/InfiniBand/EFA）的实例类型，否则网络会成为严重瓶颈
- 做 TP → 只在机器内做（NVLink），跨机器用 DP 或 PP

---

## 小结

| 并行策略 | 解决的问题 | 通信需求 | 典型使用 |
|---------|----------|---------|---------|
| 数据并行（DDP） | 训练速度 | 中（梯度同步） | 模型能放一张卡 |
| ZeRO Stage 2 | 优化器显存 | 中 | 最常用 |
| ZeRO Stage 3 / FSDP | 模型太大 | 高 | 大模型 Full FT |
| Tensor Parallelism | 模型太大 | 极高（每层都通信） | 机器内，推理为主 |
| Pipeline Parallelism | 模型太大 | 低 | 跨机器 |

对于大多数微调场景：

- **7B LoRA**：单卡搞定，不需要分布式
- **7B Full FT**：2-4 卡 + ZeRO Stage 2
- **70B LoRA**：2-4 卡 + ZeRO Stage 3
- **70B Full FT**：16+ 卡 + ZeRO Stage 3 + TP
- **预训练**：3D parallelism，这超出了本章范围

掌握了 DDP 和 ZeRO，你就能应对大多数实际训练场景了。


---

> 本章来自《LLM Infra 从入门到实践》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-llm-infra](https://github.com/diguike/book-llm-infra)
