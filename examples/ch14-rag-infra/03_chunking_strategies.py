"""
各种 Chunking 策略的对比
演示：固定长度、Recursive Character、Markdown 结构化、语义分割

运行：
  python 03_chunking_strategies.py
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chunking")


# ============================================================
# 测试文档
# ============================================================

SAMPLE_MARKDOWN = """
# LLM 推理引擎概述

大语言模型（LLM）的推理引擎是将训练好的模型部署为可服务的推理服务的核心组件。推理引擎的性能直接决定了用户体验和运营成本。

## 主流推理引擎

### vLLM

vLLM 是由 UC Berkeley 开发的高性能 LLM 推理引擎。它引入了 PagedAttention 技术，将 KV Cache 管理方式从连续内存改为分页式管理，类似于操作系统的虚拟内存机制。

主要特点：
- PagedAttention：减少 KV Cache 的内存浪费，从 60-80% 降低到 4% 以下
- Continuous Batching：动态合并不同阶段的请求，最大化 GPU 利用率
- OpenAI 兼容 API：可以作为 OpenAI API 的替代品
- 支持多种模型架构：LLaMA、Qwen、Mistral、ChatGLM 等

性能对比：在相同硬件条件下，vLLM 的吞吐量比 HuggingFace Transformers 高 14-24 倍。

### TGI (Text Generation Inference)

TGI 是 HuggingFace 开发的推理引擎。它支持 Flash Attention、Continuous Batching 等优化技术，并提供了完善的生产部署工具。

特点：
- 支持量化：GPTQ、AWQ、bitsandbytes
- 内置 Token Streaming
- 支持多 GPU Tensor Parallelism
- 提供 Docker 镜像和 Kubernetes 部署方案

### SGLang

SGLang 是由 LMSYS 团队开发的推理框架，特别擅长处理复杂的 LLM 程序（如 Tree of Thought、多步推理等）。

核心创新 RadixAttention：
- 自动检测和复用请求之间的公共前缀
- 对 few-shot prompting 场景效果显著
- 支持 constrained decoding（如 JSON 输出约束）

## 选型建议

对于大多数生产场景，推荐以下选型策略：

1. 通用推理服务：首选 vLLM，社区活跃、功能全面
2. HuggingFace 生态用户：考虑 TGI，集成度高
3. 复杂 LLM 应用：考虑 SGLang，RadixAttention 对多步推理有明显优势
4. 边缘部署：考虑 llama.cpp 或 MLC-LLM

选型时需要综合考虑模型兼容性、部署复杂度、社区支持和团队技术栈。
""".strip()


# ============================================================
# 策略 1：固定长度切分
# ============================================================

def fixed_size_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """
    固定长度切分（按字符数）。

    优点：简单、可预测
    缺点：可能在句子中间截断，语义不完整
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ============================================================
# 策略 2：Recursive Character Splitter
# ============================================================

# 按优先级排列的分隔符
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def recursive_character_split(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """
    递归字符切分——LangChain RecursiveCharacterTextSplitter 的核心逻辑。

    思路：按优先级依次尝试不同的分隔符，先尝试段落级（\\n\\n），
    不行再尝试行级（\\n），再不行按句子（。）切分...

    优点：尽量保持语义完整性
    缺点：实现稍复杂
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # 找到当前最佳分隔符
    sep = separators[-1]  # 默认用最后一个（空字符串 = 逐字符切）
    for s in separators:
        if s in text:
            sep = s
            break

    remaining_seps = separators[separators.index(sep) + 1:] if sep in separators else separators[1:]

    # 按分隔符切分
    if sep:
        parts = text.split(sep)
    else:
        parts = list(text)

    chunks = []
    current = ""

    for part in parts:
        # 拼接当前 chunk
        candidate = current + sep + part if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # 当前 chunk 已满，保存并开始新 chunk
            if current.strip():
                chunks.append(current.strip())

            # 如果单个 part 就超过 chunk_size，需要递归切分
            if len(part) > chunk_size and remaining_seps:
                sub_chunks = recursive_character_split(
                    part, chunk_size, overlap, remaining_seps
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    # 添加 overlap（从上一个 chunk 的末尾取 overlap 字符添加到下一个 chunk 的开头）
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + chunks[i])
        chunks = overlapped

    return chunks


# ============================================================
# 策略 3：基于 Markdown 结构的切分
# ============================================================

@dataclass
class MarkdownChunk:
    """Markdown chunk 包含标题上下文"""
    header: str
    content: str
    level: int  # 标题级别 (1-6)

    @property
    def text(self) -> str:
        """用于 embedding 的完整文本（包含标题以提供上下文）"""
        if self.header:
            return f"{self.header}\n\n{self.content}"
        return self.content


def markdown_chunk(
    text: str,
    max_chunk_size: int = 500,
) -> list[MarkdownChunk]:
    """
    基于 Markdown 标题结构的切分。

    思路：按 # / ## / ### 标题切分，保留标题层级作为上下文。

    优点：保持文档结构完整，每个 chunk 带有标题上下文
    缺点：只适用于 Markdown 格式
    """
    # 正则匹配 Markdown 标题
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    # 找到所有标题的位置
    headers = [(m.start(), len(m.group(1)), m.group(0)) for m in header_pattern.finditer(text)]

    if not headers:
        # 没有标题，整篇当一个 chunk
        return [MarkdownChunk(header="", content=text.strip(), level=0)]

    chunks = []
    header_stack = {}  # 用 stack 追踪各级标题

    for i, (pos, level, header_text) in enumerate(headers):
        # 确定这个 section 的内容范围
        if i + 1 < len(headers):
            next_pos = headers[i + 1][0]
        else:
            next_pos = len(text)

        content = text[pos + len(header_text):next_pos].strip()
        if not content:
            continue

        # 更新标题 stack
        header_stack[level] = header_text
        # 清除下级标题
        for l in list(header_stack.keys()):
            if l > level:
                del header_stack[l]

        # 构建完整的标题上下文（面包屑式）
        full_header = " > ".join(
            header_stack[l]
            for l in sorted(header_stack.keys())
        )

        # 如果内容过长，用 recursive_character_split 二次切分
        if len(content) > max_chunk_size:
            sub_chunks = recursive_character_split(content, max_chunk_size, overlap=0)
            for sc in sub_chunks:
                chunks.append(MarkdownChunk(
                    header=full_header,
                    content=sc,
                    level=level,
                ))
        else:
            chunks.append(MarkdownChunk(
                header=full_header,
                content=content,
                level=level,
            ))

    return chunks


# ============================================================
# 策略 4：基于句子的语义切分（简化版）
# ============================================================

def sentence_split(text: str) -> list[str]:
    """按中文/英文句子切分"""
    # 中英文标点分句
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(
    text: str,
    max_chunk_size: int = 500,
    min_chunk_size: int = 100,
) -> list[str]:
    """
    基于句子的语义切分（简化版）。

    思路：先按句子切分，然后合并相邻句子直到达到 chunk_size。
    完整的语义切分还会用 embedding 计算相邻句子的相似度来决定分割点，
    这里用简化版演示核心逻辑。

    优点：保证句子完整
    缺点：需要先做准确的分句
    """
    sentences = sentence_split(text)
    if not sentences:
        return []

    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= max_chunk_size:
            current = current + sent if current else sent
        else:
            if current and len(current) >= min_chunk_size:
                chunks.append(current)
            elif current:
                # 太短了，跟下一句合并
                current = current + sent
                continue
            current = sent

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# 对比与可视化
# ============================================================

def compare_strategies(text: str, chunk_size: int = 500):
    """对比所有 chunking 策略"""

    strategies = {
        "固定长度 (500 字, 100 overlap)": lambda: fixed_size_chunk(text, chunk_size, 100),
        "Recursive Character": lambda: recursive_character_split(text, chunk_size, 50),
        "Markdown 结构化": lambda: [c.text for c in markdown_chunk(text, chunk_size)],
        "语义（句子级）": lambda: semantic_chunk(text, chunk_size, 100),
    }

    print(f"原文长度: {len(text)} 字符\n")
    print(f"{'策略':<25} {'Chunk数':<8} {'平均长度':<10} {'最大长度':<10} {'最小长度':<10}")
    print("-" * 63)

    for name, fn in strategies.items():
        chunks = fn()
        if not chunks:
            print(f"{name:<25} 0")
            continue

        lengths = [len(c) for c in chunks]
        print(
            f"{name:<25} "
            f"{len(chunks):<8} "
            f"{sum(lengths)/len(lengths):<10.0f} "
            f"{max(lengths):<10} "
            f"{min(lengths):<10}"
        )

    # 详细展示每种策略的结果
    print("\n\n" + "=" * 70)
    for name, fn in strategies.items():
        chunks = fn()
        print(f"\n--- {name} ({len(chunks)} chunks) ---")
        for i, chunk in enumerate(chunks):
            # 截断显示
            display = chunk[:80].replace("\n", "\\n")
            if len(chunk) > 80:
                display += "..."
            print(f"  [{i}] ({len(chunk)}字) {display}")


# ============================================================
# Chunk 大小影响实验
# ============================================================

def chunk_size_experiment(text: str):
    """测试不同 chunk 大小对切分结果的影响"""
    print("\n\n" + "=" * 70)
    print("Chunk 大小影响实验（Recursive Character）")
    print("=" * 70)

    sizes = [200, 300, 500, 800, 1000]

    print(f"\n{'Chunk Size':<12} {'Chunk 数量':<12} {'平均长度':<12} {'覆盖率':<10}")
    print("-" * 46)

    for size in sizes:
        chunks = recursive_character_split(text, chunk_size=size, overlap=0)
        lengths = [len(c) for c in chunks]
        total = sum(lengths)
        coverage = total / len(text)

        print(
            f"{size:<12} "
            f"{len(chunks):<12} "
            f"{sum(lengths)/len(lengths):<12.0f} "
            f"{coverage:<10.1%}"
        )

    print("""
建议：
  - 知识库问答：300-500 字 — 检索精准度高
  - 长文摘要：800-1000 字 — 保持上下文完整
  - 代码：按函数/类切分（不适合固定长度）
""")


# ============================================================
# 主函数
# ============================================================

def main():
    print("Chunking 策略对比\n")
    compare_strategies(SAMPLE_MARKDOWN, chunk_size=500)
    chunk_size_experiment(SAMPLE_MARKDOWN)


if __name__ == "__main__":
    main()
