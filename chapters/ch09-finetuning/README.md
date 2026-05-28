# 第 9 章 微调：让通用模型变成领域专家

> **本章操作只在 Linux GPU 服务器执行。** Mac 本地可以阅读概念部分，但所有代码示例都需要 NVIDIA GPU 环境（A10/A100/RTX 4090 任意一款均可，A10 是 24GB 推理卡、A100 是 40/80GB 数据中心卡、RTX 4090 是 24GB 消费级旗舰）。QLoRA（Quantized LoRA，量化版 LoRA）部分单卡 24GB 显存够用，LoRA（Low-Rank Adaptation，低秩适配）bf16（bfloat16，2 字节浮点）至少 32GB，Full Fine-tuning（全参数微调，所有模型参数都参与训练）需要多卡。

你手上有个 7B 的通用大模型，它什么都会一点，但在你的业务场景——比如法律合同审查、医疗问诊、代码 Review——上表现平平。怎么办？

答案是微调（Fine-tuning，在预训练模型基础上用领域数据继续训练，让模型适配具体任务）。这一章从显存计算开始，搞清楚为什么 Full Fine-tuning 对大多数人不现实，然后深入 LoRA/QLoRA 这条主流路线，最后在单卡 A10 上跑通一个完整的微调流程。

## 9.1 训练循环：先建立直觉

跳进 LoRA、显存表之前，先把"训练"这件事在干什么讲清楚——后面整章的省显存技巧，都是在这个循环上做文章。

**一句话总结**：训练就是不停做这四步，直到 loss 降到满意为止。

```
forward (前向)  →  loss (算误差)  →  backward (反向)  →  optimizer.step (更新参数)
    ▲                                                              │
    └──────────────── 下一个 batch ──────────────────────────────────┘
```

**1. forward（前向传播）**：把一个 batch 的数据喂进模型，从输入一路算到输出。本质上就是第 2 章 2.1 节那张「Step 1 Tokenize → Step 5 Sampling」数据流图里走的那一遍。每一层的中间结果叫**激活值（activation）**——比如 `[batch, seq_len, 4096]` 这种张量。这些激活值不能扔，因为下面反向传播要用。

**2. loss（损失）**：拿模型输出和"正确答案"对比，算出一个标量数字。预测越准，这个数字越小。SFT 用的损失函数是交叉熵，下面 9.5 节专门讲。先把它理解成「模型这一步有多错」的总分就够。

**3. backward（反向传播）**：从 loss 反着用链式求导法则，一路算回每个参数对 loss 的贡献。算完之后，每个参数 `p` 上都挂了一个 `p.grad`——这就是**梯度（gradient）**，一个和 `p` 形状完全一样的张量。

**梯度到底是什么**：`p.grad[i]` 告诉你「`p[i]` 这个参数往哪边动、能让 loss 下降多少」。JS 工程师可以这样类比一步参数更新：

```js
params[i] -= learning_rate * grad[i]
//            ↑ 每步迈多大      ↑ 往哪边走、走多猛
```

70 亿个参数，每个都按自己的 grad 调一点点——这就是一步训练。跑几千几万步，loss 就慢慢降下来。

**4. optimizer.step（参数更新）**：拿着每个参数的梯度，按算法更新参数。最常用的优化器是 Adam（以及它的变体 AdamW），它额外为每个参数维护两个 fp32 状态——梯度的一阶矩 `m`（移动平均）和二阶矩 `v`（梯度平方的移动平均）——用来让更新方向更稳。这两个状态就是下面 9.2 节那张显存表里 60.8 GB 的大头。

PyTorch 代码就是这四步的字面翻译：

```python
for batch in dataloader:
    out  = model(batch)               # 1. forward
    loss = criterion(out, labels)     # 2. loss
    loss.backward()                   # 3. backward，自动算所有参数的梯度
    optimizer.step()                  # 4. update
    optimizer.zero_grad()             # 清掉梯度，准备下一轮
```

### 训练为什么这么占显存

理解了这四步，下面 9.2 节那张「Full FT 要 95 GB」的显存表就有了来历：

| 显存大头 | 来源 | 大小（7B 模型） |
|---------|------|---------------|
| 模型参数 | forward 一直要用 | 15 GB（bf16） |
| 激活值 | forward 保存下来留给 backward | 2-8 GB |
| 梯度 | backward 算出来，每个参数一份 | 15 GB（bf16） |
| 优化器状态 | Adam 给每个参数额外维护 m、v 两个 fp32 | 60 GB |

所以这一章接下来看到的每一个省显存技巧，本质都是在这四样东西上做文章：

- **LoRA**：原始权重冻结、不算梯度，只对新增的小矩阵算 → 「梯度」「优化器状态」从 7.6B × N 降到 80M × N，砍掉 99%
- **QLoRA**：原始权重再做 4-bit 量化 → 「模型参数」从 15 GB 压到 ~4 GB
- **gradient checkpointing**：forward 只保留少数检查点，backward 时重新跑一段 → 「激活值」省 30-60%，代价是多算一次 forward
- **gradient accumulation**：连续 N 步只累加梯度不更新 → 等效放大 batch size 但显存不增

记住这张「四件套」表，后面 9.2-9.6 节那些"为什么这样能省显存"的说法就都顺了。

## 9.2 Full Fine-tuning vs PEFT

### Full Fine-tuning 的显存账

先算一笔账。拿 Qwen2-7B（7.6B 参数）为例，Full Fine-tuning 时显存占用分这几块：

| 组成部分 | 计算方式 | 显存占用 |
|---------|---------|---------|
| 模型参数（bf16） | 7.6B × 2 bytes | 15.2 GB |
| 梯度（bf16） | 7.6B × 2 bytes | 15.2 GB |
| Adam 优化器状态 | 7.6B × 2 × 4 bytes（fp32 的一阶矩 m + 二阶矩 v） | 60.8 GB |
| 激活值（batch_size=1） | 取决于序列长度，约 | 2-8 GB |
| **总计** | | **~95 GB** |

一张 A100 80GB 都装不下。就算用 gradient checkpointing 省掉一部分激活值，至少也要两张 A100。

> **bf16 vs fp16**：bf16（bfloat16，2 字节）和 fp16（half precision，2 字节半精度浮点）等大，但数值范围更宽（指数位 8 bit，和 fp32 即 32 位单精度浮点一样），不容易溢出，训练更稳定。代价是有效精度低于 fp16。Ampere（NVIDIA 第 8 代 GPU 架构，A10/A100）及以上架构都原生支持 bf16，本书后面默认用 bf16。

最大头是 Adam 优化器——每个参数要额外存两个 fp32 状态（一阶矩 `m`、二阶矩 `v`），直接占了 60 GB+。7B 模型尚且如此，70B 就更不用想。

**关于 gradient checkpointing**：9.1 节讲过 backward 算梯度需要用到 forward 时的激活值，所以默认情况下激活值要一直留在显存里。gradient checkpointing 只保留少数"检查点"位置的激活值，需要时重新跑一段 forward 算回中间层。典型效果是激活值显存省 30-60%，代价是训练速度多花 ~20%——显存吃紧时几乎是必开项。

### 为什么 Full FT 对大多数人不现实

不仅仅是硬件成本的问题：

1. **显存需求高**：7B Full FT 要 ~95GB，70B 要 ~950GB，需要多卡并行
2. **训练时间长**：7B 在单卡 A100 上（如果能装下）跑完一个 epoch（遍历整个训练集一次的过程）要好几个小时
3. **灾难性遗忘（catastrophic forgetting）**：全量更新参数容易把模型原有的通用能力搞坏——比如微调成法律助手后，模型可能忘记怎么写代码、解数学题
4. **存储成本**：每个微调任务都要保存一份完整的模型权重，7B 就是 15GB

对于有 8 卡 A100 集群的大厂来说，Full FT 当然可以做。但对于绝大多数团队，我们需要更实际的方案。

### PEFT：只训练极少量参数

PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）的核心思想很简单：冻结原始模型的绝大部分参数，只训练新增的少量参数。

主流的 PEFT 方法包括：

- **LoRA/QLoRA**：在冻结的权重矩阵旁边加低秩分解矩阵，最主流
- **Prefix Tuning**（前缀微调）：在输入前加可训练的虚拟 token（virtual token，不对应真实词表的占位向量，只在训练时学习）
- **Adapter**（适配器）：在 Transformer 层之间插入小型网络
- **IA3**（Infused Adapter by Inhibiting and Amplifying Inner Activations）：用极少量参数缩放注意力（attention）和前馈层（FFN，feed-forward network）的激活值

其中 LoRA 是目前绝对的主流，后面我们重点讲它。

## 9.3 LoRA / QLoRA

> 原始论文：LoRA [arxiv 2106.09685](https://arxiv.org/abs/2106.09685)；QLoRA [arxiv 2305.14314](https://arxiv.org/abs/2305.14314)。PEFT 库实现：[github.com/huggingface/peft](https://github.com/huggingface/peft)（[官方文档](https://huggingface.co/docs/peft)）。

### LoRA 原理：低秩分解

LoRA（Low-Rank Adaptation，低秩适配）的思路很直接。低秩分解（low-rank decomposition）指把一个大矩阵拆成两个"瘦长矩阵"相乘，秩 r 远小于原矩阵的行列数。原始模型的某个权重矩阵 $W \in \mathbb{R}^{d \times k}$（神经网络里一层的参数，输入维度 k、输出维度 d），在微调时我们不直接更新 $W$，而是在旁边加一个低秩分解：

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

结构上长这样：

```
       输入 x (维度 k)
        │
        ├──────────────┐
        ▼              ▼
   ┌─────────┐    ┌─────────┐
   │  W      │    │  A      │  (r × k)
   │ (d × k) │    └────┬────┘
   │ 冻结     │         │ 中间维度 r
   └────┬────┘         ▼
        │         ┌─────────┐
        │         │  B      │  (d × r)
        │         │ 可训练   │
        │         └────┬────┘
        │              │
        ▼              ▼
        └─────  +  ─────┘
                │
                ▼
            输出 y (维度 d)
```

原始权重 $W$ 全程冻结，只有旁路上的 $A$ 和 $B$ 两个小矩阵参与训练。推理时把 $BA$ 加回 $W$ 得到 $W' = W + BA$，结构完全不变，没有任何额外开销。

举个具体例子。Qwen2-7B 的 `q_proj`（attention 的 Query 投影，每层有 q/k/v/o 四个投影，第 2 章 2.1 节有讲）权重矩阵是 $4096 \times 4096$，有 16.8M 个参数。如果 LoRA rank（秩，低秩矩阵的中间维度，越大表达能力越强但参数也越多）设为 16，那 $B$ 是 $4096 \times 16$，$A$ 是 $16 \times 4096$，加起来只有 131K 个参数——不到原始的 1%。

### LoRA 三个核心超参：rank / alpha / target_modules

**rank（r）**：LoRA 矩阵的秩，最重要的超参数。

- `r=8`：最常用的起步值，适合简单任务
- `r=16`：更好的效果，大多数场景的甜蜜点
- `r=32-64`：复杂任务或追求最佳效果时使用
- `r=128+`：接近 Full FT 的效果，但训练成本也上去了

**alpha（lora_alpha）**：缩放系数，最终的 LoRA 贡献是 $\frac{\alpha}{r} \times BA$。

- 经验法则：`alpha = 2 × rank`
- 比如 `rank=16, alpha=32`

**target_modules**：对哪些层加 LoRA。

```python
# 最常见：只对 attention 的 q/v 投影加 LoRA
# q_proj / k_proj / v_proj / o_proj 分别是 Query/Key/Value/Output 投影
# gate_proj / up_proj / down_proj 是 FFN 里 SwiGLU 结构的三个线性层
target_modules = ["q_proj", "v_proj"]

# 更好的效果：对所有线性层加 LoRA
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

加的层越多，可训练参数越多，效果通常越好，但显存和训练时间也相应增加。实践中，对所有线性层加 LoRA（`target_modules="all-linear"`）往往是性价比最高的选择。

### QLoRA：4-bit 量化 + LoRA，单卡就能微调 7B

QLoRA 的核心动作就一句话：**把基础模型量化到 4-bit（每个权重只用 4 位存）再上 LoRA**。听起来激进，但实测精度损失大多在 1-2% 以内——原因在于它把"高精度"和"省显存"分到两条通道里。

**为什么 4-bit 量化训练却不掉点**

- **基础模型 4-bit 量化后全程冻结**，只参与 forward 算激活值，**不算梯度、不更新**——所以量化误差不会被梯度循环放大
- **LoRA adapter 全程保持 bf16**，所有真正的学习信号都在高精度通道上传播。基模负责"我原来是什么"，adapter 负责"我新学了什么"
- **矩阵乘法时临时反量化**：4-bit 权重在算 `out = x @ W` 之前会临时反量化（dequantize）回 bf16，算完结果丢弃，下一次再反量化。整个训练里"4-bit 只是存储格式，计算精度还是 bf16"

QLoRA 原论文（[arxiv 2305.14314](https://arxiv.org/abs/2305.14314)）在多个基准上验证了这个结论——对绝大多数应用任务来说，1-2% 的损失换 4 倍显存是划算的。

**三个工程细节**

- **NF4**（NormalFloat 4-bit）：针对神经网络权重的实际分布（接近正态分布）设计的非线性 4-bit 格式。比直接均匀量化精度更高。代码里设 `bnb_4bit_quant_type="nf4"`，直接用就行。
- **Double Quantization**（双重量化）：量化时每 64 个权重共享一组缩放常数，这些常数本身还要再量化一次。多省约 0.4 GB——不大，但白送的，对应 `bnb_4bit_use_double_quant=True`。
- **bitsandbytes**（简称 bnb）：Tim Dettmers 写的开源量化库，HuggingFace 的官方 4-bit/8-bit 量化后端。和 `transformers`、`peft` 无缝集成，是 QLoRA 的事实实现。装它 `pip install bitsandbytes`。

**显存对比**（Qwen2-7B，rank=16，所有线性层）：

| 方法 | 模型参数 | 可训练参数 | 优化器状态 | 总显存 |
|-----|---------|----------|----------|-------|
| Full FT (bf16) | 15.2 GB | 15.2 GB | 60.8 GB | ~95 GB |
| LoRA (bf16) | 15.2 GB | ~160 MB | ~640 MB | ~18 GB |
| QLoRA (4-bit) | ~4 GB | ~160 MB | ~640 MB | ~7 GB |

QLoRA 把 7B 模型的微调门槛降到了消费级 GPU——24GB 的 RTX 4090 / 3090 跑得很从容，16GB 的卡（4080）牺牲点 batch size 也能跑。第一次做微调的工程师，QLoRA 几乎是默认起点。

**几个实际使用上的坑**

- 只有 Ampere 及以上的卡（A10 / A100 / RTX 30/40 系）能用 bf16，老的 Turing 架构（RTX 20 系、2080Ti）要退回 fp16，数值稳定性差一些
- 训练时 `bnb_4bit_compute_dtype` 不能从 bf16 改成 fp16，容易出 NaN
- 合并 LoRA 权重时**不能在 4-bit 状态下合并**，要重新加载基础模型成 bf16 再合并——见 9.6 节 Step 3

## 9.4 数据准备

微调的效果，50% 取决于数据质量。这一节按工程师的实际工作流走：**先看数据长什么样（格式）→ 从哪里来（来源）→ 怎么判断好坏（质量）→ 怎么洗（清洗）**。

### 常见数据格式

微调数据其实就是「指令 → 期望回答」的成对样本。这类训练范式叫指令微调（Instruction Tuning，给模型喂大量「指令-回答」样本，让它学会按用户指令生成内容）。下面三种是最常见的存储格式。

**Alpaca 格式**（Stanford Alpaca 项目最早使用的 JSON 字段格式，最常见）：

```json
{
  "instruction": "将以下英文翻译成中文",
  "input": "The weather is nice today.",
  "output": "今天天气很好。"
}
```

**ShareGPT 格式**（来自 ShareGPT 网站的 ChatGPT 对话分享数据，原生支持多轮对话）：

```json
{
  "conversations": [
    {"from": "human", "value": "帮我写一个 Python 快排"},
    {"from": "gpt", "value": "好的，以下是快速排序的实现...\n```python\ndef quicksort(arr):\n    ..."},
    {"from": "human", "value": "能加个注释吗？"},
    {"from": "gpt", "value": "当然，以下是带注释的版本...\n```python\ndef quicksort(arr):\n    # ..."}
  ]
}
```

**OpenAI 格式**（OpenAI Chat Completions API 使用的 messages 数组结构，role 字段固定为 system/user/assistant，已成事实标准）：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个法律助手"},
    {"role": "user", "content": "什么是竞业禁止条款？"},
    {"role": "assistant", "content": "竞业禁止条款是指..."}
  ]
}
```

### 数据从哪里来

最常用的几种方式：

1. **人工标注**：质量最高，成本也最高，适合核心场景的种子数据
2. **GPT-4 生成**：给 GPT-4 一些示例让它扩写，质量不错，注意合规和成本
3. **从已有数据转换**：把内部文档、FAQ、客服记录、工单转成指令-回答对，往往是最被忽视的高 ROI 来源
4. **Self-Instruct**（自生成指令，用一小批种子指令引导 LLM 不断生成新指令的方法）：模型自己生成指令再人工筛选（论文：[arxiv 2212.10560](https://arxiv.org/abs/2212.10560)）

实际项目的常用组合：**先人工标注 200-500 条高质量种子 → 用 GPT-4 扩展到 2000-5000 条 → 最后人工审核一轮**。这套打法比纯人工省钱、比纯 GPT-4 生成可靠。

### 数据质量 > 数据数量

这一点有大量实证支持：

- **LIMA 论文**（Less Is More for Alignment，Meta 2023 提出"对齐少即是多"，[arxiv 2305.11206](https://arxiv.org/abs/2305.11206)）：只用 1000 条精心挑选的数据，就能让 LLaMA-65B（Meta 的 650 亿参数开源大模型）达到接近 GPT-4（OpenAI 当时最强的闭源模型）的对话质量
- **Stanford Alpaca**（斯坦福基于 LLaMA 微调的开源指令模型项目，[github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)）：52K 条 GPT-3.5（OpenAI ChatGPT 背后的模型）生成的数据，效果就相当不错
- 实际经验：**500-2000 条高质量数据 > 50000 条低质量数据**

判断"高质量"的 5 个维度：

1. **准确性**：答案必须正确，这是底线
2. **完整性**：回答覆盖了问题的关键方面
3. **格式一致性**：所有样本遵循相同的回答风格和格式
4. **多样性**：覆盖目标场景的不同类型问题
5. **难度分布**：简单、中等、困难的问题都要有

### 数据清洗的 5 个步骤

实际项目中，你拿到的原始数据往往很脏。下面 5 步按顺序做一遍，能洗掉 80% 的常见问题：

1. **去重**：用 MinHash（基于哈希的近似集合相似度算法，能快速找出"几乎相同"的样本）或 exact match 精确去重。重复数据会让模型过拟合（overfitting，训练集表现好但泛化差）到特定模式。开源库 `datasketch` 直接能用。
2. **长度过滤**：太短的回答通常质量差。比如 `data = [d for d in data if len(d["output"]) > 50]`，截断阈值看场景调。
3. **格式检查**：比如需要 JSON 输出的任务，用 `json.loads()` 试解一遍，挡掉非法 JSON 样本。
4. **用 LLM 打分**：把样本喂给 GPT-4 让它对质量打 1-5 分，保留 ≥4 分的。成本可控（1 美元能扫几千条），比人工快得多。
5. **人工抽查**：最后随机抽 100 条人眼看一遍——LLM 打分会漏掉一些靠常识才能识别的问题（比如答非所问但写得通顺、事实硬伤）。

每一步的真实代码见 `examples/ch09-finetuning/data/clean_pipeline.py`。

## 9.5 训练框架

懂了 LoRA / QLoRA 的原理之后，剩下的就是选个工具把训练跑起来。常见的训练栈有三种，从最贴底层到最自动化：

| 方案 | 写代码 | 速度 / 显存 | 谁该选 |
|-----|------|-----------|-------|
| 方案一：transformers + peft 徒手写 | 多 | 一般 | 想看清每一步、做自定义改造 |
| 方案二：LLaMA-Factory | 写 YAML | 一般 | 大多数微调任务（SFT/DPO/PPO 全覆盖） |
| 方案三：Unsloth | 5 行起步 | 最快、最省 | 单卡 QLoRA 追性价比 |

下面先把训练目标讲清楚（这是所有方案共通的底层），再分别看三种方案的写法。

### SFT 的训练目标：next-token prediction

看代码之前先把训练目标说清楚。SFT 让模型做一件事：**给定一段文本，正确预测下一个 token**——这就是 next-token prediction，从预训练到微调，自回归语言模型一直用的就是这个目标。

把 prompt 和 response 拼成一条完整序列后，模型逐位置预测下一个 token，对预测出的概率分布和真实 token 算**交叉熵**（cross-entropy，分类任务最常用的损失，衡量两个概率分布的差距），公式 $-\log P(\text{真实 token})$，对所有位置求和取平均。这就是 9.1 节里说的那个 loss。

关键细节：**prompt 部分不计 loss，只有 response 部分计梯度**。实现上是把 `labels` 里 prompt 位置全填 `-100`——PyTorch 的交叉熵约定遇到 `-100` 就跳过这个位置。不做这一步，模型会去学"生成 prompt"，但我们要的是「看到 prompt → 生成 response」，不是反过来。

### 方案一：徒手用 transformers + peft（看清每一步）

最基础的训练栈：`transformers` 加载模型，`peft` 套上 LoRA。这是最贴近底层的写法，适合想看清每一步在干什么的读者。核心代码就这几步：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
# AutoModelForCausalLM：自动加载 GPT 类自回归模型（即"因果语言模型 Causal LM"）
from trl import SFTTrainer, SFTConfig  # 用 trl 的 SFTTrainer，不要直接用 transformers.Trainer，原因见下面 callout
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# 0. 加载数据（详细切训/验集见 9.6 Step 1）
ds = load_dataset("json", data_files="data/sample_train.jsonl", split="train")
ds = ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = ds["train"], ds["test"]

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 83,886,080 || all params: 7,699,898,368 || trainable%: 1.09

# 3. 训练
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,                  # 训练 3 个 epoch
    per_device_train_batch_size=4,       # 每张卡每步喂入的样本数
    gradient_accumulation_steps=4,       # 梯度累积，等效 batch size 放大 N 倍
    learning_rate=2e-4,                  # 学习率，对应 9.1 节里参数更新公式里的 lr
    lr_scheduler_type="cosine",          # 学习率调度：cosine 余弦退火，先升后按余弦曲线衰减
    warmup_ratio=0.1,                    # 预热：开头 10% 步数把 lr 从 0 缓慢爬到目标值，避免一上来就震荡
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,
    max_seq_length=2048,                 # 单条样本最大 token 数，超过会截断
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)
trainer.train()
```

> **为什么用 SFTTrainer 而不是普通 Trainer？** 普通 `transformers.Trainer` 不知道哪段是 prompt、哪段是 response，会对整个序列连同 padding（为了把不等长样本对齐而填的占位 token）一起算 loss，方向就跑偏了。`trl.SFTTrainer` 自带 prompt 的 label mask 和 padding 处理，开箱即用。如果一定要用普通 Trainer，得自己写 data collator——把一个 batch 的样本拼成可输入模型的张量的"打包函数"——把 prompt 和 padding 位置的 labels 设为 `-100`。TRL 库：[github.com/huggingface/trl](https://github.com/huggingface/trl)（[文档](https://huggingface.co/docs/trl)）。

完整代码见 `examples/ch09-finetuning/01_lora_from_scratch.py`。

### 方案二：LLaMA-Factory（一份 YAML 跑起来）

如果你不想写那么多代码，[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 是目前最好的选择——一站式微调框架，把数据加载、训练、合并、评估全打包成命令行 + Web UI，所有常见微调场景都封装成 YAML 配置：

```yaml
### model
model_name_or_path: Qwen/Qwen2-7B
template: qwen

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all

### dataset
dataset: my_custom_data
cutoff_len: 2048

### output
output_dir: ./output/qwen2-7b-lora

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
flash_attn: fa2     # FlashAttention v2，省显存的 attention 实现，详见 8.1 节
gradient_checkpointing: true
```

然后一行命令开训：

```bash
llamafactory-cli train config.yaml
```

LLaMA-Factory 支持 100+ 模型、LoRA/QLoRA/Full FT、SFT 和 DPO/PPO/ORPO 等多种偏好对齐方式（这些算法第 10 章会讲），还内置了 Web UI。大多数微调任务，直接用它最省事。

### 方案三：Unsloth（单卡 QLoRA 追极致速度）

[Unsloth](https://github.com/unslothai/unsloth) 是高效微调加速库，号称比 HuggingFace PEFT 快 2 倍、省 60% 显存。它的做法是用 Triton（OpenAI 开源的 GPU kernel DSL，比直接写 CUDA 简单）手写了 LoRA 的前向和反向 kernel，连带优化了 RoPE、cross-entropy、4-bit 量化等热点路径，所以在单卡 QLoRA 场景下性能领先比较明显。

**什么时候选 Unsloth：**

- 单卡（A10 / RTX 4090 / 3090）做 QLoRA，想又快又省显存
- 不需要多卡分布式——Unsloth 免费版只支持单卡，多卡要付费 Pro 版
- 接受 Unsloth 自家维护的算法封装范围（目前主要是 LoRA、QLoRA、DPO、ORPO，覆盖大多数微调场景）

**5 行代码起步：**

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2-7B",     # Unsloth 在 HuggingFace 上预转换好的模型
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32)
```

后面接 `SFTTrainer` 训练的代码和方案一一样，把 `model` 换成 Unsloth 包过的版本即可。

### 三种方案怎么选

| 维度 | 方案一 transformers + peft | 方案二 LLaMA-Factory | 方案三 Unsloth |
|------|------------------------|------------------|-------------|
| 上手成本 | 高（写代码） | 低（写 YAML） | 低（5 行代码） |
| 性能 | 一般 | 一般 | 单卡 QLoRA 最快 |
| 自定义灵活度 | 最高 | 中（YAML 能配的范围内） | 中 |
| 多卡 | 自己接 accelerate / deepspeed | 内置 | 免费版不支持 |
| 算法覆盖 | 全部（自己写就行） | SFT / DPO / PPO / ORPO 全覆盖 | LoRA / QLoRA / DPO / ORPO |
| 适合谁 | 做研究 / 写自定义 loss / 改训练逻辑 | 大多数微调场景，第一选择 | 单卡 QLoRA 追性价比 |

实践路径推荐：**第一次跑通用 LLaMA-Factory**——配置最少、坑最少；**调参试错用 Unsloth**——同样的硬件能多跑几轮实验；**写不进 YAML 的自定义需求回到 transformers + peft 徒手写**——可控性最强。

### 训练超参速查：每个怎么调、彼此怎么联动

**learning_rate**：LoRA 通常用 `1e-4` 到 `5e-4`，QLoRA 用 `2e-4` 是常见起点。Full FT 要小很多，通常 `1e-5` 到 `5e-5`。

**num_train_epochs**：微调不需要太多 epoch。1-3 个 epoch 通常就够了。数据量小（<1000 条）时可以多跑几个 epoch（3-5），但要注意过拟合。

**per_device_train_batch_size × gradient_accumulation_steps**：有效 batch size = `batch_size × accumulation_steps × gpu_count`，指的是一次参数更新覆盖的样本总数。通常 16-64 是合理范围。

**gradient_accumulation_steps**：每隔 N 步才做一次参数更新，把这 N 步的梯度累加起来。效果等价于把 batch size 放大 N 倍，但每一步的显存占用没有增加。显存吃紧但又想要大 batch size 时必开。

**gradient_checkpointing**：见 9.2 节的说明，反向传播时不保存全部激活值，用时重新前向计算。显存节省 30-60%，训练速度损失约 20%。LoRA/QLoRA 训练几乎是默认开启的项。

**lora_dropout**：dropout 是经典正则化手段——训练时按概率随机把一部分神经元输出置零，强迫模型别过度依赖某几个权重。`lora_dropout` 只对 LoRA 矩阵开 dropout。小数据集（< 5000 条）设 0.05-0.1，数据量大时设为 0。

**warmup_ratio**：建议 0.05-0.1，让学习率在开始时缓慢上升。

### 训练时怎么看 Loss 曲线（异常了改什么）

训练时最重要的监控指标就是 loss 曲线。把 `logging_steps` 设成 10-20，配合 TensorBoard 或 [Weights & Biases](https://wandb.ai)（业界主流的 ML 实验跟踪平台，把 loss / eval / 显存等指标可视化）实时看曲线。

下面 5 种典型形态对照一下，遇到对应症状直接照方抓药——**看见异常信号停掉训练改参数，比硬跑完更省时间**。

| 现象 | 大概率原因 | 改什么 |
|-----|----------|-------|
| **正常**：快下降 → 缓下降 → 趋于平稳 | — | 继续训。3 epoch 后 loss 在 0.8-1.2 是 SFT 的健康区间 |
| **过拟合**：train loss 一直降，eval loss 开始上升 | 数据少 / epoch 太多 / rank 太大 | 减 epoch；加数据；调大 `lora_dropout`（0.05 → 0.1）；减小 rank |
| **loss 剧烈波动甚至发散为 NaN** | 学习率太大 / 没开 bf16 | 把 `learning_rate` 砍一半（2e-4 → 1e-4）；确认 `bf16=True`；QLoRA 场景检查 `bnb_4bit_compute_dtype=bf16` |
| **loss 几乎不动** | 学习率太小 / 还在 warmup 阶段 | 等过完 warmup（默认 10% 步数）再判断；过完仍然不动就把 lr 调大 2-5 倍 |
| **loss 早早卡住**（稳定在 1.5+ 降不动） | 数据质量差 / chat template 错 / mask 没生效 | 抽 20 条样本人工看；确认 prompt 部分被 mask 成 `-100`；确认用了模型对应的 chat template（Qwen2 / Llama-3 各不同） |

**几个数字记忆点（指令微调 SFT 场景）：**

- 起始 loss 通常在 2-3 之间
- 第 1 个 epoch 末降到 1.0-1.5
- 第 3 个 epoch 末稳定在 0.8-1.2 是健康的
- 降到 0.3 以下：大概率过拟合，模型已经在死记答案
- 降到 0.05 以下：基本可确定过拟合（除非数据极度简单且重复）

LoRA 训练多跑一小时都在烧电费——肯花 5 分钟看一眼 loss 曲线，往往比死等 3 个 epoch 跑完再发现问题省得多。

## 9.6 实战：用 QLoRA 微调 Qwen2-7B

这一节在单卡 A10（24GB）上跑通一个完整的微调流程，4 步打通从原始数据到上线服务：

```
Step 1 数据准备       Step 2 训练 LoRA       Step 3 合并权重        Step 4 部署
─────────────       ──────────────       ──────────────       ─────────────
切训/验集            QLoRA 4-bit           adapter 合回基模      vLLM serve
让 SFTTrainer       单卡 A10 跑 3 epoch     得到普通 HF 模型      OpenAI 兼容 API
自动 tokenize       约 30 分钟              ./output/merged       :8000 端口
```

每一步的原理上面 9.1-9.5 已经讲过，这一节把它们串成一条可运行的流水线。

### Step 1：数据准备（切训/验集 + Tokenize 时机）

数据这一步在工程上有几个细节值得讲清楚，不然实战时会卡。

**数据文件用 JSONL（每行一条 JSON）**：

```json
{"messages": [{"role": "user", "content": "什么是梯度下降？"}, {"role": "assistant", "content": "梯度下降是一种优化算法..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

JSONL 的好处是流式读取——`datasets` 库一行行扫不会一次性把所有数据加载进内存，几 GB 的文件也扛得住。

**切训练 / 验证集**：用 `datasets` 库读 JSONL，按 9:1 切：

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="data/sample_train.jsonl", split="train")
ds = ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = ds["train"], ds["test"]
```

验证集用来算 eval loss（上一节 Loss 曲线里那个监控过拟合的指标）。10% 是默认拆分比例，数据少时（< 500 条）可以拆 20%，数据多时（> 10k）可以拆 5%。`seed=42` 是为了可复现——同一份数据每次切出来的训/验集都一样。

**Tokenize 谁来做？** `SFTTrainer` 会自动用模型对应的 tokenizer 把 `messages` 转成 token ID 张量，**不需要手动 tokenize**。它内部会按模型的 chat template（不同模型的对话格式约定，Qwen2 / Llama-3 / DeepSeek 各自不同）把 system / user / assistant 拼成完整字符串，再调 tokenizer。这也是用 `SFTTrainer` 而不是普通 `Trainer` 的核心原因——chat template 拼接和 prompt mask 都帮你做了。

**数据规模参考**：

- 跑通流程：50-200 条就够
- 单一狭窄任务（特定格式输出、特定领域问答）：500-2000 条精挑细选
- 多任务通用助手风格：5000-50000 条混合数据

完整示例文件见 `examples/ch09-finetuning/data/sample_train.jsonl`。

### Step 2：QLoRA 训练

核心配置 + 模型加载 + LoRA 包装，三段拼起来：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# (1) 4-bit 量化配置（BitsAndBytesConfig 来自 bitsandbytes 库，QLoRA 默认的量化后端）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                         # 把模型权重加载为 4 位
    bnb_4bit_quant_type="nf4",                 # 用 NF4 格式量化
    bnb_4bit_compute_dtype=torch.bfloat16,     # 实际矩阵乘法时反量化到 bf16
    bnb_4bit_use_double_quant=True,            # 启用双重量化进一步省显存
)

# (2) 用 bnb_config 加载 4-bit 模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    quantization_config=bnb_config,            # ← 关键：把量化配置传进来
    torch_dtype=torch.bfloat16,
    device_map="auto",                         # 自动分配到可用 GPU
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

# (3) 套 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 83,886,080 || all params: 7,699,898,368 || trainable%: 1.09
```

后面的 `SFTConfig` / `SFTTrainer` 训练代码和 9.5 方案一基本一样，把 `model` 换成上面这个 4-bit 包装过的版本即可。训练大约 30 分钟跑完 3 个 epoch（1000 条数据），显存占用约 6-7 GB，A10 绰绰有余。

完整代码见 `examples/ch09-finetuning/02_qlora_train.py`。

### Step 3：合并 LoRA 权重

训练完成后，LoRA adapter 是单独保存的（通常只有几十 MB）。如果要用 vLLM（高吞吐 LLM 推理引擎）部署，需要先把 adapter 合并回基础模型：

```python
from peft import PeftModel    # PeftModel：peft 库里把基础模型 + adapter 合并管理的类
from transformers import AutoModelForCausalLM

# 加载基础模型（全精度）
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B", torch_dtype=torch.bfloat16
)

# 加载 LoRA adapter
model = PeftModel.from_pretrained(base_model, "./output/qwen2-7b-qlora")

# 合并权重
merged_model = model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./output/qwen2-7b-merged")
```

完整代码见 `examples/ch09-finetuning/03_merge_lora.py`。

### Step 4：用 vLLM 部署

合并后的模型就是一个标准的 HuggingFace 模型（HuggingFace Hub 兼容的目录结构，含 config.json + tokenizer + 权重文件），直接用 vLLM 部署：

```bash
vllm serve ./output/qwen2-7b-merged \
    --tensor-parallel-size 1 \   # Tensor Parallel（张量并行，跨多卡切分单层矩阵）的卡数
    --max-model-len 4096 \
    --port 8000
```

然后用标准的 OpenAI API（OpenAI Chat Completions HTTP 接口规范，已成行业事实标准）格式访问：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./output/qwen2-7b-merged",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 同样的流程，LLaMA-Factory 版本

上面 4 步把 transformers + peft 的底层流程跑了一遍——好处是看清每一步在做什么。同样的流程换成 LLaMA-Factory 大致只需要一份 YAML + 两条命令：

```yaml
# config.yaml — 涵盖 Step 1 数据加载 + Step 2 训练 + Step 3 合并的配置
model_name_or_path: Qwen/Qwen2-7B
quantization_bit: 4              # 等于 QLoRA
finetuning_type: lora
lora_target: all
dataset: my_data                 # 在 dataset_info.json 里登记好数据路径
output_dir: ./output/qwen2-7b-qlora
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
num_train_epochs: 3
bf16: true
gradient_checkpointing: true
val_size: 0.1                    # 自动切 10% 做验证集
```

```bash
# 训练 + 自动合并 adapter
llamafactory-cli train config.yaml

# 导出合并后的模型
llamafactory-cli export config_export.yaml

# 部署
vllm serve ./output/qwen2-7b-qlora-merged --max-model-len 4096
```

完整配置文件见 `examples/ch09-finetuning/04_llamafactory_config.yaml`。

**第一次微调建议徒手跑一遍上面的 Step 1-4**，搞清楚每一步在做什么；熟悉之后切到 LLaMA-Factory 提效；做实验阶段 Unsloth 跑得快。三种方案打配合，比死守一种舒服。

---

## 小结

| 方法 | 可训练参数占比 | 7B 显存需求 | 适用场景 |
|-----|-------------|----------|---------|
| Full FT | 100% | ~95 GB | 有大量 GPU 资源 |
| LoRA | ~1% | ~18 GB | 有 A100/A10 |
| QLoRA | ~1% | ~7 GB | 消费级 GPU |

对于大多数实际项目，QLoRA 是最佳起点。等到效果不够好、需要进一步优化时，再考虑 LoRA（bf16）或 Full FT。

下一章我们讲对齐——怎么让微调后的模型不仅"能力强"，而且"回答得好"。


---

> 本章来自《LLM Infra 从入门到实践》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-llm-infra](https://github.com/diguike/book-llm-infra)
