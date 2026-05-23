# 第 2 章 代码示例

## 环境准备

```bash
# 进入示例目录
cd examples/ch02-transformer

# 安装依赖
pip install -r requirements.txt
```

## 硬件要求

| 示例 | 最低要求 | 推荐 |
|------|---------|------|
| 01-03 | CPU | - |
| 04 | GPU 8GB VRAM | RTX 3090 / 阿里云 ecs.gn7i-c8g1.2xlarge |

## 示例列表

### 01_tokenizer_compare.py
对比三种主流 Tokenizer 的分词结果和词表大小。

```bash
python 01_tokenizer_compare.py --text "你好，我正在学习 LLM 基础设施"
```

### 02_attention_visualize.py
加载预训练模型，可视化 Attention 权重热力图。

```bash
python 02_attention_visualize.py --model gpt2 --text "The cat sat on the mat"
```

### 03_kv_cache_demo.py
对比有无 KV Cache 时的推理速度差异。

```bash
python 03_kv_cache_demo.py --seq-length 256
```

### 04_nano_gpt_train.py
基于 nanoGPT 简化版，在 tinyShakespeare 数据集上训练一个字符级 GPT。

```bash
# 训练（首次会自动下载 ~1.1 MB 数据集；CPU 约 8 分钟跑完默认 20 epochs）
python 04_nano_gpt_train.py --train

# 生成文本
python 04_nano_gpt_train.py --generate --prompt "ROMEO:"

# 一键训练 + 生成
python 04_nano_gpt_train.py --train --generate

# 用自定义文本训练
python 04_nano_gpt_train.py --train --data-file my_text.txt
```

离线环境下数据集下载失败时会自动退化到一段内置中文短文本（仅供验证流程）。
