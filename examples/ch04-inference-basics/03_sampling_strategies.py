"""
第 4 章 示例 3：实现各种采样策略并对比输出

从头实现 Greedy、Temperature、Top-k、Top-p、Repetition Penalty，
对比它们在相同 prompt 下的不同输出。
"""

import torch
import torch.nn.functional as F
from typing import Optional, Set
from dataclasses import dataclass


@dataclass
class SamplingParams:
    """采样参数"""
    temperature: float = 1.0
    top_k: int = 0          # 0 = 不使用 top-k
    top_p: float = 1.0      # 1.0 = 不使用 top-p
    repetition_penalty: float = 1.0
    max_tokens: int = 100
    seed: Optional[int] = 42


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Temperature scaling.
    T < 1: 分布更尖锐 (更确定性)
    T = 1: 不变
    T > 1: 分布更平坦 (更随机)
    """
    if temperature == 0:
        return logits  # temperature=0 时用 greedy
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k 过滤: 只保留概率最高的 k 个 token
    """
    if k <= 0:
        return logits

    # 找到 top-k 的阈值
    top_k_values, _ = torch.topk(logits, min(k, logits.size(-1)))
    threshold = top_k_values[..., -1]

    # 低于阈值的设为 -inf
    logits = logits.clone()
    logits[logits < threshold] = float('-inf')
    return logits


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Top-p (Nucleus) 过滤: 保留累积概率达到 p 的最小 token 集合
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 找到累积概率超过 p 的位置（但保留第一个 token）
    sorted_mask = cumulative_probs - sorted_probs > p
    sorted_logits[sorted_mask] = float('-inf')

    # 恢复原始顺序
    logits = logits.clone()
    logits.scatter_(-1, sorted_indices, sorted_logits)
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: Set[int],
    penalty: float,
) -> torch.Tensor:
    """
    Repetition penalty: 降低已生成 token 的概率

    实现逻辑:
    - 如果 logit > 0: 除以 penalty (降低)
    - 如果 logit < 0: 乘以 penalty (更负)
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    logits = logits.clone()
    for token_id in generated_ids:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def sample_token(logits: torch.Tensor, params: SamplingParams,
                 generated_ids: Optional[Set[int]] = None) -> int:
    """
    完整的采样流程: logits → token_id

    Pipeline:
    1. Repetition penalty
    2. Temperature scaling
    3. Top-k filtering
    4. Top-p filtering
    5. 采样或 argmax
    """
    if generated_ids is None:
        generated_ids = set()

    # 取最后一个位置的 logits
    if logits.dim() > 1:
        logits = logits[-1]

    # 1. Repetition penalty
    logits = apply_repetition_penalty(logits, generated_ids, params.repetition_penalty)

    # 2. Greedy (temperature = 0)
    if params.temperature == 0:
        return torch.argmax(logits).item()

    # 3. Temperature
    logits = apply_temperature(logits, params.temperature)

    # 4. Top-k
    logits = apply_top_k(logits, params.top_k)

    # 5. Top-p
    logits = apply_top_p(logits, params.top_p)

    # 6. 转概率并采样
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate_with_sampling(
    model,
    tokenizer,
    prompt: str,
    params: SamplingParams,
    device: str = "cuda",
) -> str:
    """使用自定义采样策略生成文本"""
    if params.seed is not None:
        torch.manual_seed(params.seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    generated_ids = set()

    with torch.no_grad():
        # Prefill
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1, :]

        all_tokens = input_ids[0].tolist()

        for _ in range(params.max_tokens):
            token_id = sample_token(logits, params, generated_ids)
            all_tokens.append(token_id)
            generated_ids.add(token_id)

            if token_id == tokenizer.eos_token_id:
                break

            # Decode step
            next_input = torch.tensor([[token_id]], device=device)
            outputs = model(next_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]

    return tokenizer.decode(all_tokens, skip_special_tokens=True)


def visualize_temperature_effect():
    """可视化 Temperature 对概率分布的影响"""
    print("\n" + "=" * 60)
    print("Temperature 对概率分布的影响")
    print("=" * 60)

    # 模拟 5 个 token 的 logits
    logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.1])
    token_names = ["The", "A", "This", "That", "My"]

    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

    print(f"\n原始 logits: {logits.tolist()}")
    print(f"\n{'Token':<8}", end="")
    for t in temperatures:
        print(f"{'T=' + str(t):<12}", end="")
    print()
    print("-" * 68)

    for i, name in enumerate(token_names):
        print(f"{name:<8}", end="")
        for t in temperatures:
            scaled = logits / t
            probs = F.softmax(scaled, dim=-1)
            bar_len = int(probs[i].item() * 30)
            bar = "█" * bar_len
            print(f"{probs[i].item():.3f} {bar:<7}", end="")
        print()


def visualize_top_p_effect():
    """可视化 Top-p 的动态截断"""
    print("\n" + "=" * 60)
    print("Top-p Nucleus Sampling 动态截断")
    print("=" * 60)

    # 尖锐分布
    logits_sharp = torch.tensor([5.0, 2.0, 1.0, 0.5, 0.1, -1.0, -2.0])
    # 平坦分布
    logits_flat = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    names = ["token_0", "token_1", "token_2", "token_3", "token_4", "token_5", "token_6"]

    for label, logits in [("尖锐分布", logits_sharp), ("平坦分布", logits_flat)]:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        print(f"\n{label}:")
        print(f"  {'Token':<10} {'Prob':<8} {'CumProb':<10} {'Top-p=0.9':<12}")
        print(f"  {'-' * 38}")
        for i in range(len(names)):
            idx = sorted_indices[i].item()
            included = "  ✓ 保留" if cumsum[i] - sorted_probs[i] < 0.9 else "  ✗ 丢弃"
            print(f"  {names[idx]:<10} {sorted_probs[i].item():.4f}   "
                  f"{cumsum[i].item():.4f}    {included}")


def main():
    # 先展示采样策略的可视化（不需要模型）
    visualize_temperature_effect()
    visualize_top_p_effect()

    # 用实际模型对比不同采样策略
    print("\n" + "=" * 60)
    print("不同采样策略的输出对比")
    print("=" * 60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("需要安装 transformers: pip install transformers torch")
        return

    model_name = "Qwen/Qwen2-0.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n加载模型: {model_name} ({device})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    prompt = "Once upon a time, in a land far away,"

    strategies = {
        "Greedy (T=0)": SamplingParams(temperature=0, max_tokens=60),
        "Low Temp (T=0.3)": SamplingParams(temperature=0.3, max_tokens=60),
        "Medium Temp (T=0.7)": SamplingParams(temperature=0.7, top_p=0.9, max_tokens=60),
        "High Temp (T=1.5)": SamplingParams(temperature=1.5, max_tokens=60),
        "Top-k=10": SamplingParams(temperature=0.8, top_k=10, max_tokens=60),
        "Top-p=0.5": SamplingParams(temperature=0.8, top_p=0.5, max_tokens=60),
        "With RepPenalty=1.3": SamplingParams(
            temperature=0.7, top_p=0.9, repetition_penalty=1.3, max_tokens=60
        ),
    }

    for name, params in strategies.items():
        print(f"\n--- {name} ---")
        print(f"  参数: T={params.temperature}, top_k={params.top_k}, "
              f"top_p={params.top_p}, rep_penalty={params.repetition_penalty}")
        output = generate_with_sampling(model, tokenizer, prompt, params, device)
        # 只打印生成部分
        generated = output[len(prompt):].strip()
        print(f"  输出: {generated[:200]}")


if __name__ == "__main__":
    main()
