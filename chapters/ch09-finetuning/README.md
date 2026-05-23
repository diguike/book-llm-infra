# 第 9 章 微调：让通用模型变成领域专家

> **本章操作只在 Linux GPU 服务器执行。** Mac 本地可以阅读概念部分，但所有代码示例都需要 NVIDIA GPU 环境（A10/A100/RTX 4090 任意一款均可，A10 是 24GB 推理卡、A100 是 40/80GB 数据中心卡、RTX 4090 是 24GB 消费级旗舰）。QLoRA（Quantized LoRA，量化版 LoRA，ch01 已介绍）部分单卡 24GB 显存够用，LoRA（Low-Rank Adaptation，低秩适配，ch01 已介绍）bf16（bfloat16，2 字节浮点，ch03 已介绍）至少 32GB，Full Fine-tuning（全参数微调，所有模型参数都参与训练）需要多卡。

你手上有个 7B 的通用大模型，它什么都会一点，但在你的业务场景——比如法律合同审查、医疗问诊、代码 Review——上表现平平。怎么办？

答案是微调（Fine-tuning，在预训练模型基础上用领域数据继续训练，让模型适配具体任务）。这一章从显存计算开始，搞清楚为什么 Full Fine-tuning 对大多数人不现实，然后深入 LoRA/QLoRA 这条主流路线，最后在单卡 A10 上跑通一个完整的微调流程。

## 9.1 Full Fine-tuning vs PEFT

### Full Fine-tuning 的显存账

先算一笔账。拿 Qwen2-7B（7.6B 参数）为例，Full Fine-tuning 时显存占用分这几块：

| 组成部分 | 计算方式 | 显存占用 |
|---------|---------|---------|
| 模型参数（bf16） | 7.6B × 2 bytes | 15.2 GB |
| 梯度（gradient，反向传播算出来的参数更新方向，bf16） | 7.6B × 2 bytes | 15.2 GB |
| Adam 优化器（optimizer，按梯度更新参数的算法；Adam 是最常用的自适应优化器）状态 | 7.6B × 2 × 4 bytes（fp32 的一阶矩 m 和二阶矩 v） | 60.8 GB |
| 激活值（activation，前向传播中每层的中间输出，反向传播算梯度时需要用到；batch_size=1 指一次喂入 1 条样本） | 取决于序列长度，约 | 2-8 GB |
| **总计** | | **~95 GB** |

一张 A100 80GB 都装不下。就算用 gradient checkpointing（梯度检查点，用计算时间换显存的优化技术，下面会展开）省掉一部分激活值，你还是需要至少两张 A100。

> **bf16 vs fp16**：bf16（bfloat16，2 字节）和 fp16（half precision，2 字节半精度浮点）等大，但数值范围更宽（指数位 8 bit，和 fp32 即 32 位单精度浮点一样），不容易溢出，训练更稳定。代价是有效精度低于 fp16。Ampere（NVIDIA 第 8 代 GPU 架构，A10/A100）及以上架构都原生支持 bf16，本书后面默认用 bf16。

这里的关键是 Adam 优化器。它为每个参数维护两个 fp32 的状态（一阶矩 m 是梯度的移动平均，二阶矩 v 是梯度平方的移动平均），直接占了 60GB+。7B 模型如此，70B 模型就更不用想了。

**关于 gradient checkpointing**：反向传播（backward，从 loss 反向算每个参数的梯度的过程，对应前向传播 forward）时本来要保存每一层的激活值用来算梯度，gradient checkpointing 只保存少数几个"检查点"，需要时重新前向计算一遍中间层。典型效果是激活值显存节省 30-60%，训练速度损失约 20%——典型的"时间换空间"策略，显存吃紧的时候几乎是必开项。

### 为什么 Full FT 对大多数人不现实

不仅仅是硬件成本的问题：

1. **显存需求高**：7B Full FT 要 ~95GB，70B 要 ~950GB，需要多卡并行
2. **训练时间长**：7B 在单卡 A100 上（如果能装下）跑完一个 epoch（遍历整个训练集一次的过程）要好几个小时
3. **灾难性遗忘（catastrophic forgetting）**：全量更新参数很容易把模型在通用任务上的能力搞坏
4. **存储成本**：每个微调任务都要保存一份完整的模型权重，7B 就是 15GB

对于有 8 卡 A100 集群的大厂来说，Full FT 当然可以做。但对于绝大多数团队，我们需要更实际的方案。

### PEFT：只训练极少量参数

PEFT（Parameter-Efficient Fine-Tuning，参数高效微调，ch01 已介绍）的核心思想很简单：冻结原始模型的绝大部分参数，只训练新增的少量参数。

主流的 PEFT 方法包括：

- **LoRA/QLoRA**：在冻结的权重矩阵旁边加低秩分解矩阵，最主流
- **Prefix Tuning**（前缀微调）：在输入前加可训练的虚拟 token（virtual token，不对应真实词表的占位向量，只在训练时学习）
- **Adapter**（适配器）：在 Transformer 层之间插入小型网络
- **IA3**（Infused Adapter by Inhibiting and Amplifying Inner Activations）：用极少量参数缩放注意力（attention，ch02 已介绍）和前馈层（FFN，feed-forward network，ch02 已介绍）的激活值

其中 LoRA 是目前绝对的主流，后面我们重点讲它。

## 9.2 LoRA / QLoRA

> 原始论文：LoRA [arxiv 2106.09685](https://arxiv.org/abs/2106.09685)；QLoRA [arxiv 2305.14314](https://arxiv.org/abs/2305.14314)。PEFT 库实现：[github.com/huggingface/peft](https://github.com/huggingface/peft)（[官方文档](https://huggingface.co/docs/peft)）。

### LoRA 原理：低秩分解

LoRA（Low-Rank Adaptation，低秩适配）的思路非常优雅。低秩分解（low-rank decomposition）指把一个大矩阵拆成两个"瘦长矩阵"相乘，秩 r 远小于原矩阵的行列数。原始模型的某个权重矩阵 $W \in \mathbb{R}^{d \times k}$（神经网络里一层的参数，输入维度 k、输出维度 d），在微调时我们不直接更新 $W$，而是在旁边加一个低秩分解：

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

举个具体例子。Qwen2-7B 的 `q_proj`（attention 里把输入投影成 Query 向量的线性层，Transformer 每层都有 q/k/v/o 四个投影）权重矩阵是 $4096 \times 4096$，有 16.8M 个参数。如果 LoRA rank（秩，低秩矩阵的中间维度，越大表达能力越强但参数也越多）设为 16，那 $B$ 是 $4096 \times 16$，$A$ 是 $16 \times 4096$，加起来只有 131K 个参数——不到原始的 1%。

推理时，可以把 LoRA 矩阵合并回原始权重：$W' = W + BA$，不增加任何推理开销。

### 关键超参数

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

### QLoRA：量化 + LoRA

QLoRA（Quantized LoRA）的核心创新是：把基础模型量化到 4-bit（每个权重只用 4 位存储，ch07 已介绍量化基础），然后在量化模型上做 LoRA。

具体来说：

1. 用 NF4（NormalFloat 4-bit，针对正态分布权重设计的 4 位浮点格式，ch07 已介绍）量化基础模型，显存从 15GB 降到 ~4GB
2. LoRA 的 adapter（适配器，挂在原模型旁边的小型可训练矩阵）矩阵保持 bf16 精度
3. 计算时，量化权重反量化（dequantize，把低位权重还原回高精度浮点）到 bf16 做矩阵乘法（matmul，神经网络的核心运算）
4. 梯度只更新 LoRA 参数（bf16）
5. 用 Double Quantization（双重量化，对量化常数本身再做一次量化）额外节省约 0.4 GB 显存，对应代码里的 `bnb_4bit_use_double_quant=True`（bnb 是 bitsandbytes 库的简称，QLoRA 的官方量化实现）

显存对比（Qwen2-7B，rank=16，所有线性层）：

| 方法 | 模型参数 | 可训练参数 | 优化器状态 | 总显存 |
|-----|---------|----------|----------|-------|
| Full FT (bf16) | 15.2 GB | 15.2 GB | 60.8 GB | ~95 GB |
| LoRA (bf16) | 15.2 GB | ~160 MB | ~640 MB | ~18 GB |
| QLoRA (4-bit) | ~4 GB | ~160 MB | ~640 MB | ~7 GB |

QLoRA 让你在一张消费级 RTX 4090 (24GB) 甚至 RTX 3090 上就能微调 7B 模型。这意味着个人开发者也能在自己的机器上微调大模型了。

效果上，QLoRA 和 LoRA 的差距在大多数任务上很小（1-2% 以内），但显存节省非常显著。

## 9.3 数据准备

微调的效果，50% 取决于数据质量。

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

### 数据质量 > 数据数量

这一点有大量实证支持：

- **LIMA 论文**（Less Is More for Alignment，Meta 2023 提出"对齐少即是多"，[arxiv 2305.11206](https://arxiv.org/abs/2305.11206)）：只用 1000 条精心挑选的数据，就能让 LLaMA-65B（Meta 的 650 亿参数开源大模型）达到接近 GPT-4（OpenAI 当时最强的闭源模型）的对话质量
- **Stanford Alpaca**（斯坦福基于 LLaMA 微调的开源指令模型项目，[github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)）：52K 条 GPT-3.5（OpenAI ChatGPT 背后的模型）生成的数据，效果就相当不错
- 实际经验：**500-2000 条高质量数据 > 50000 条低质量数据**

什么是"高质量"？

1. **准确性**：答案必须正确，这是底线
2. **完整性**：回答覆盖了问题的关键方面
3. **格式一致性**：所有样本遵循相同的回答风格和格式
4. **多样性**：覆盖目标场景的不同类型问题
5. **难度分布**：简单、中等、困难的问题都要有

### 数据清洗技巧

实际项目中，你拿到的原始数据往往很脏。几个实用的清洗步骤：

```python
# 1. 去重：用 MinHash（基于哈希的近似集合相似度算法，能快速找出"几乎相同"的样本）
#    或 exact match（精确字符串匹配）去重
# 重复数据会让模型过拟合（overfitting，在训练集上表现好但泛化差）到特定模式

# 2. 长度过滤：太短的回答通常质量差
data = [d for d in data if len(d["output"]) > 50]

# 3. 格式检查：确保数据格式正确
# 比如需要 JSON 输出的任务，检查输出是否是合法 JSON

# 4. 用 LLM 打分：用 GPT-4 对数据质量打分
# 保留评分 > 4/5 的样本

# 5. 人工抽查：随机抽 100 条看看质量
```

### 构造指令微调数据的方法

最常用的几种方式：

1. **人工标注**：质量最高，成本也最高。适合核心场景的种子数据
2. **GPT-4 生成**：给 GPT-4 一些示例，让它生成更多。质量不错，注意合规
3. **从已有数据转换**：把文档、FAQ、客服记录转成指令-回答对
4. **Self-Instruct**（自生成指令，用一小批种子指令引导 LLM 不断生成新指令的方法）：让模型自己生成指令，然后人工筛选（论文：[arxiv 2212.10560](https://arxiv.org/abs/2212.10560)）

实际项目推荐组合策略：先人工标注 200-500 条高质量种子数据，然后用 GPT-4 扩展到 2000-5000 条，最后人工审核一轮。

## 9.4 训练框架

### SFT 的训练目标：next-token prediction

在看代码之前先把训练目标说清楚。SFT（Supervised Fine-Tuning，监督微调，用「输入-期望输出」成对样本训练，ch01 已介绍）的损失函数（loss function，衡量预测与真实标签差距的函数，训练目标就是把它降到最小）是 **next-token prediction（下一个 token 预测，自回归语言模型的基本训练目标）的交叉熵损失**（cross-entropy loss，分类任务最常用的损失函数，衡量预测概率分布和真实分布的差距）：给定 prompt（提示词，模型的输入部分）+response（响应，模型要生成的部分）拼成的完整序列，模型一个 token（最小词元，ch02 已介绍）一个 token 地预测下一个 token 的概率分布，损失定义为 $-\log P(\text{真实 token})$，对序列里所有 token 求和取平均。

关键细节：**prompt 部分的 token 不计入 loss**，只有 response 部分计梯度。这通过 `labels` 张量（tensor，PyTorch 里的多维数组，类比 JS 里的嵌套数组但支持 GPU 加速）里把 prompt 位置设成 `-100`（PyTorch（Meta 开源的深度学习框架，本书训练侧的默认框架）交叉熵的 ignore_index，约定值为 -100 的位置不参与损失计算）来实现。如果不做这个 mask（掩码，把某些位置标记为"不参与计算"），模型会去拟合用户的输入，这显然不是我们想要的——我们要的是"看到 prompt 生成 response"，不是"看到 prompt 生成另一个 prompt"。

### HuggingFace Transformers + PEFT

HuggingFace（业界最大的开源模型托管社区 + 一整套配套库）的 `transformers` 库是模型加载/训练的事实标准，`peft` 是它的官方参数高效微调库，类比 Node 生态里的 `express` + 中间件。最基础的方式，适合想深入理解细节的同学。核心代码就这几步：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# AutoModelForCausalLM：自动加载因果语言模型（Causal LM，自回归预测下一个 token 的模型，GPT 系全是这类）
# AutoTokenizer：自动加载对应模型的分词器
from trl import SFTTrainer, SFTConfig  # trl（Transformer Reinforcement Learning，HuggingFace 的对齐/微调训练库）
# 注意：用 trl 的 SFTTrainer，不要直接用 Trainer
from peft import LoraConfig, get_peft_model, TaskType
# peft：HuggingFace 官方 PEFT 库；LoraConfig 是 LoRA 超参数；get_peft_model 把普通模型包成 PEFT 模型

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
    per_device_train_batch_size=4,       # batch_size：每张卡每步喂入的样本数
    gradient_accumulation_steps=4,       # 梯度累积，等效 batch size 放大 N 倍
    learning_rate=2e-4,                  # 学习率（learning rate），梯度下降每步迈多大
    lr_scheduler_type="cosine",          # 学习率调度器（lr scheduler），cosine 即余弦退火，先升后按余弦曲线衰减
    warmup_ratio=0.1,                    # warmup（预热），训练开始时学习率从 0 缓慢爬升，避免一上来就震荡
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,
    max_seq_length=2048,                 # 单条样本最大 token 数，超过会截断
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

> **为什么用 SFTTrainer 而不是普通 Trainer？** 普通 `transformers.Trainer` 不知道哪部分是 prompt、哪部分是 response，会对整个序列计 loss（包括 padding token，即为了把短样本对齐到同一长度而填充的占位 token），这会让模型学到错误的目标。`trl.SFTTrainer` 内置了 prompt 部分的 labels masking 和 padding 处理。如果一定要用普通 Trainer，需要自己写 data collator（数据整理器，负责把一批样本组装成可输入模型的 tensor，比如做 padding 对齐和构造 labels），把 prompt 部分和 padding 位置的 labels 设为 `-100`。TRL 库：[github.com/huggingface/trl](https://github.com/huggingface/trl)（[文档](https://huggingface.co/docs/trl)）。

完整代码见 `examples/ch09-finetuning/01_lora_from_scratch.py`。

### LLaMA-Factory：一站式微调框架

如果你不想写那么多代码，[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)（开源的一站式微调框架，把数据加载、训练、合并、评估全打包成命令行 + Web UI）是目前最好的选择。它把所有常见的微调场景封装成了 YAML（一种缩进式的配置文件格式，比 JSON 适合人手写）配置：

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
flash_attn: fa2     # FlashAttention v2，IO 友好的高效 attention 实现，可大幅省显存提速度
gradient_checkpointing: true
```

然后一行命令开训：

```bash
llamafactory-cli train config.yaml
```

LLaMA-Factory 支持 100+ 模型、LoRA/QLoRA/Full FT、SFT/DPO（Direct Preference Optimization，直接偏好优化，无需训练奖励模型的偏好对齐方法，ch01 已介绍）/PPO（Proximal Policy Optimization，近端策略优化，RLHF 经典的强化学习算法，ch10 会详讲）/ORPO（Odds Ratio Preference Optimization，比值比偏好优化，SFT 和偏好对齐一步完成）等多种训练方式，还内置了 Web UI。对于大多数微调任务，推荐直接用它。

### Unsloth

[Unsloth](https://github.com/unslothai/unsloth) — 高效微调加速库，号称比 HuggingFace PEFT 快 2 倍、省 60% 显存。通过手写 Triton（OpenAI 开源的 Python DSL，用来写 GPU kernel，比直接写 CUDA 简单）kernel（GPU 上跑的核心计算函数）优化了 LoRA 的前向和反向传播。如果你在单卡上做 QLoRA 微调，Unsloth 是值得尝试的选择。安装：`pip install unsloth`。

### 关键训练参数

**learning_rate**：LoRA 通常用 `1e-4` 到 `5e-4`，QLoRA 用 `2e-4` 是常见起点。Full FT 要小很多，通常 `1e-5` 到 `5e-5`。

**num_train_epochs**：微调不需要太多 epoch。1-3 个 epoch 通常就够了。数据量小（<1000 条）时可以多跑几个 epoch（3-5），但要注意过拟合。

**per_device_train_batch_size × gradient_accumulation_steps**：有效 batch size（一次参数更新所基于的样本总数）= batch_size × accumulation_steps × gpu_count。通常 16-64 是合理范围。

**gradient_accumulation_steps**：每隔 N 步才做一次参数更新，把这 N 步的梯度累加起来。效果等价于把 batch size 放大 N 倍，但每一步的显存占用没有增加。显存吃紧但又想要大 batch size 时必开。

**gradient_checkpointing**：见 9.1 节的说明，反向传播时不保存全部激活值，用时重新前向计算。显存节省 30-60%，训练速度损失约 20%。LoRA/QLoRA 训练几乎是默认开启的项。

**lora_dropout**：dropout 是一种正则化技术，训练时按概率随机置零一部分神经元的输出，强迫模型不要过度依赖某些权重。`lora_dropout` 就是只对 LoRA 矩阵的前向传播开启 dropout，防止过拟合。小数据集（< 5000 条）设 0.05-0.1；数据量大时可以设为 0。

**warmup_ratio**：建议 0.05-0.1，让学习率在开始时缓慢上升。

### Loss 曲线解读

训练过程中最重要的监控指标就是 loss（损失值，对应前面说的 cross-entropy loss 的实时数值）曲线：

- **正常曲线**：快速下降 → 缓慢下降 → 趋于平稳
- **过拟合信号**：train loss（训练集上的损失）持续下降，但 eval loss（evaluation loss，验证集上的损失，反映模型在没见过的数据上的表现）开始上升
- **学习率太大**：loss 剧烈波动，甚至发散（出现 NaN，浮点数的"非法值"标记，意味着数值溢出）
- **学习率太小**：loss 下降非常慢，几乎不动
- **数据质量问题**：loss 很快降到一个值就不再下降了

实际经验：对于指令微调任务（SFT），训练 2-3 个 epoch 后 loss 降到 0.8-1.2 左右是正常的。如果 loss 降到 0.3 以下，大概率过拟合了。

## 9.5 实战：用 QLoRA 微调 Qwen2-7B

这一节我们在单卡 A10 (24GB) 上完整跑通一个微调流程。

### Step 1：准备数据

我们准备一个简单的指令微调数据集（中文问答场景），格式如下：

```json
{"messages": [{"role": "user", "content": "什么是梯度下降？"}, {"role": "assistant", "content": "梯度下降是一种优化算法..."}]}
```

详见 `examples/ch09-finetuning/data/sample_train.jsonl`。

### Step 2：QLoRA 训练

核心配置：

```python
# 4-bit 量化配置（BitsAndBytesConfig 来自 bitsandbytes 库，QLoRA 默认的量化后端）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                         # 把模型权重加载为 4 位
    bnb_4bit_quant_type="nf4",                 # 用 NF4 格式量化
    bnb_4bit_compute_dtype=torch.bfloat16,     # 实际矩阵乘法时反量化到 bf16
    bnb_4bit_use_double_quant=True,            # 启用双重量化进一步省显存
)

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
```

训练大约 30 分钟可以跑完 3 个 epoch（1000 条数据）。显存占用约 6-7GB，A10 绰绰有余。

完整代码见 `examples/ch09-finetuning/02_qlora_train.py`。

### Step 3：合并 LoRA 权重

训练完成后，LoRA adapter 是单独保存的（通常只有几十 MB）。如果要用 vLLM（高吞吐 LLM 推理引擎，ch01/ch05 已介绍）部署，需要先把 adapter 合并回基础模型：

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
    --tensor-parallel-size 1 \   # Tensor Parallel（张量并行，跨多卡切分单层矩阵，ch03 已介绍）的卡数
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

### 或者：用 LLaMA-Factory 一步到位

如果用 LLaMA-Factory，整个流程更简单。配置文件见 `examples/ch09-finetuning/04_llamafactory_config.yaml`，一行命令搞定训练。

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
