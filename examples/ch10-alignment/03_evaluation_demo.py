"""
模型评估 Demo

演示三种评估方式：
1. lm-evaluation-harness 跑标准 benchmark
2. LLM-as-Judge 自动评估
3. 简单的人工评估框架

使用方法:
    # 跑 benchmark（需要安装 lm-eval）
    python 03_evaluation_demo.py --mode benchmark --model ./my-model

    # LLM-as-Judge 评估
    python 03_evaluation_demo.py --mode judge --model ./my-model

    # 生成人工评估用的对比数据
    python 03_evaluation_demo.py --mode human --model-a ./model-a --model-b ./model-b
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ============================================================
# 方式一：lm-evaluation-harness 标准 Benchmark
# ============================================================
def run_benchmark(model_path: str, tasks: str = "mmlu,hellaswag,arc_challenge"):
    """
    用 lm-evaluation-harness 跑标准 benchmark。

    安装: pip install lm-eval
    文档: https://github.com/EleutherAI/lm-evaluation-harness

    常用任务:
    - mmlu: 57 学科多选题，衡量知识广度
    - hellaswag: 常识推理补全
    - arc_challenge: 科学多选题
    - truthfulqa_mc2: 真实性评估
    - gsm8k: 数学应用题
    - humaneval: 代码生成（需要额外配置）
    """
    output_dir = "./eval_results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16",
        "--tasks", tasks,
        "--batch_size", "8",
        "--output_path", output_dir,
        "--log_samples",
    ]

    print(f"Running benchmark evaluation:")
    print(f"  Model: {model_path}")
    print(f"  Tasks: {tasks}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
        print(f"\nResults saved to {output_dir}")
    except FileNotFoundError:
        print("Error: lm_eval not found. Install with: pip install lm-eval")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running lm_eval: {e}")
        sys.exit(1)


# ============================================================
# 方式二：LLM-as-Judge
# ============================================================

# 评估 prompt 模板
JUDGE_PROMPT = """请评估以下 AI 助手的回答质量，从 1-5 分打分。

评分标准：
- 5 分：准确、完整、格式清晰、非常有帮助
- 4 分：基本正确，有小瑕疵，整体有帮助
- 3 分：部分正确，有明显遗漏或不准确之处
- 2 分：大部分不正确或不相关
- 1 分：完全错误、有害或拒绝回答合理问题

用户问题：
{question}

AI 回答：
{answer}

请严格按以下 JSON 格式输出，不要输出其他内容：
{{"score": <1-5>, "reason": "<简短评价原因>"}}"""


def llm_as_judge(model_path: str, eval_data_path: str = None):
    """
    用一个强力 LLM（如 GPT-4）评估目标模型的回答质量。

    流程：
    1. 用目标模型对评估集生成回答
    2. 用 GPT-4 对每个回答打分
    3. 统计平均分和分布
    """
    # 示例评估数据
    eval_questions = [
        "什么是 KV Cache？为什么它对 LLM 推理很重要？",
        "解释一下 LoRA 微调的原理",
        "数据并行和模型并行有什么区别？",
        "什么是量化？有哪些常见的量化方法？",
        "如何评估一个微调后模型的质量？",
    ]

    print("=" * 60)
    print("LLM-as-Judge Evaluation Pipeline")
    print("=" * 60)

    # Step 1: 用目标模型生成回答
    print(f"\nStep 1: Generate answers with {model_path}")
    print("(In production, you would load the model and generate here)")

    # 示例：模拟生成回答
    # 实际使用时，用 transformers 或 vLLM 生成
    sample_answers = [
        "KV Cache 缓存了之前 token 的 Key 和 Value，避免重复计算...",
        "LoRA 在冻结的权重矩阵旁边加一个低秩分解矩阵...",
        "数据并行是每张卡一份模型副本，模型并行是把模型切分到多卡...",
        "量化是把模型参数从高精度转到低精度，常见方法有 GPTQ、AWQ...",
        "可以从训练指标、Benchmark、LLM-as-Judge、人工评估多个层面...",
    ]

    # Step 2: 构造评估 prompt
    print("\nStep 2: Construct judge prompts")
    judge_prompts = []
    for q, a in zip(eval_questions, sample_answers):
        prompt = JUDGE_PROMPT.format(question=q, answer=a)
        judge_prompts.append(prompt)

    # Step 3: 调用 GPT-4 评分
    print("\nStep 3: Call GPT-4 for scoring")
    print("(In production, call OpenAI API here)")

    # 示例代码（需要 OPENAI_API_KEY）：
    print("""
    # 实际调用代码：
    from openai import OpenAI
    client = OpenAI()

    scores = []
    for prompt in judge_prompts:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        scores.append(result)

    avg_score = sum(s["score"] for s in scores) / len(scores)
    print(f"Average score: {avg_score:.2f}")
    """)


# ============================================================
# 方式三：人工评估框架
# ============================================================
def generate_human_eval_data(
    model_a_path: str,
    model_b_path: str,
    output_path: str = "./human_eval_data.jsonl",
):
    """
    生成 A/B 测试用的人工评估数据。

    输出格式：每行包含 prompt + 两个模型的回答（随机顺序），
    让标注员选择更好的回答。
    """
    import random

    eval_questions = [
        "什么是梯度下降？",
        "解释 Transformer 的核心组件",
        "LoRA 和全量微调有什么区别？",
        "DPO 和 RLHF 有什么区别？",
        "什么是分布式训练？为什么需要它？",
    ]

    print(f"Generating human evaluation data...")
    print(f"  Model A: {model_a_path}")
    print(f"  Model B: {model_b_path}")

    # 实际使用时，加载两个模型分别生成回答
    # 这里用占位符示意
    eval_items = []
    for i, question in enumerate(eval_questions):
        # 随机决定 A/B 显示顺序（消除位置偏见）
        show_a_first = random.random() > 0.5
        item = {
            "id": i,
            "question": question,
            "response_1": f"[Model {'A' if show_a_first else 'B'} response]",
            "response_2": f"[Model {'B' if show_a_first else 'A'} response]",
            "response_1_model": "A" if show_a_first else "B",
            "response_2_model": "B" if show_a_first else "A",
            # 标注员填写以下字段
            "preference": None,       # "response_1" or "response_2" or "tie"
            "annotator": None,
            "notes": None,
        }
        eval_items.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Human evaluation data saved to {output_path}")
    print(f"Total items: {len(eval_items)}")
    print(f"\nNext steps:")
    print(f"  1. Replace placeholder responses with actual model outputs")
    print(f"  2. Distribute to 3+ annotators")
    print(f"  3. Compute inter-annotator agreement (Cohen's Kappa)")
    print(f"  4. Compute win rate for each model")


def analyze_human_eval(eval_data_path: str):
    """分析人工评估结果"""
    results = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line.strip()))

    # 统计胜率
    model_a_wins = 0
    model_b_wins = 0
    ties = 0

    for item in results:
        pref = item.get("preference")
        if pref is None:
            continue
        if pref == "tie":
            ties += 1
        elif pref == "response_1":
            winner = item["response_1_model"]
        else:
            winner = item["response_2_model"]

        if pref != "tie":
            if winner == "A":
                model_a_wins += 1
            else:
                model_b_wins += 1

    total = model_a_wins + model_b_wins + ties
    if total > 0:
        print(f"\nHuman Evaluation Results ({total} samples):")
        print(f"  Model A wins: {model_a_wins} ({model_a_wins/total*100:.1f}%)")
        print(f"  Model B wins: {model_b_wins} ({model_b_wins/total*100:.1f}%)")
        print(f"  Ties: {ties} ({ties/total*100:.1f}%)")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation demo")
    parser.add_argument(
        "--mode", type=str, choices=["benchmark", "judge", "human"],
        default="judge", help="Evaluation mode"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--model-a", type=str, default=None)
    parser.add_argument("--model-b", type=str, default=None)
    parser.add_argument(
        "--tasks", type=str, default="mmlu,hellaswag,arc_challenge",
        help="Benchmark tasks (comma-separated)"
    )
    args = parser.parse_args()

    if args.mode == "benchmark":
        run_benchmark(args.model, args.tasks)
    elif args.mode == "judge":
        llm_as_judge(args.model)
    elif args.mode == "human":
        model_a = args.model_a or args.model
        model_b = args.model_b or args.model
        generate_human_eval_data(model_a, model_b)
