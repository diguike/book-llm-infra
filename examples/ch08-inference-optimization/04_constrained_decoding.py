"""结构化输出 / 约束解码演示

约束解码在 logit 层面屏蔽非法 token，保证输出 100% 符合指定格式。
原理: JSON Schema → 正则表达式 → 有限状态机 (FSM) → 每步 token mask。

使用方法:
    # FSM 原理演示 (不需要 GPU 或模型)
    python 04_constrained_decoding.py

    # 用 outlines 做约束解码 (需要 GPU)
    python 04_constrained_decoding.py --outlines --model Qwen/Qwen2-0.5B
"""

import argparse
import json


def demo_fsm_principle():
    """演示 FSM 驱动的约束解码原理"""

    print("=" * 65)
    print("约束解码原理: FSM (有限状态机) 驱动的 Token Masking")
    print("=" * 65)
    print()
    print('目标: 输出合法 JSON，格式 {"name": <string>, "age": <integer>}')
    print()

    # 模拟 FSM 的状态转换
    steps = [
        {
            "state": "OBJECT_START",
            "generated": "",
            "allowed": '{ 开头',
            "next_token": "{",
            "blocked_examples": "a, [, 123",
        },
        {
            "state": "EXPECT_KEY_1",
            "generated": "{",
            "allowed": '"name"',
            "next_token": '"name"',
            "blocked_examples": '"age", "foo", 123',
        },
        {
            "state": "EXPECT_COLON",
            "generated": '{"name"',
            "allowed": ":",
            "next_token": ":",
            "blocked_examples": '=, {, "',
        },
        {
            "state": "EXPECT_STRING",
            "generated": '{"name":',
            "allowed": '" (字符串开头)',
            "next_token": ' "',
            "blocked_examples": "123, true, {",
        },
        {
            "state": "IN_STRING",
            "generated": '{"name": "',
            "allowed": "任意字符 或 \" (结束)",
            "next_token": 'Alice"',
            "blocked_examples": "} (字符串未闭合)",
        },
        {
            "state": "EXPECT_COMMA",
            "generated": '{"name": "Alice"',
            "allowed": ",",
            "next_token": ",",
            "blocked_examples": '} (还有 age 未填)',
        },
        {
            "state": "EXPECT_KEY_2",
            "generated": '{"name": "Alice",',
            "allowed": '"age"',
            "next_token": ' "age"',
            "blocked_examples": '"name", "x"',
        },
        {
            "state": "EXPECT_COLON_2",
            "generated": '{"name": "Alice", "age"',
            "allowed": ":",
            "next_token": ":",
            "blocked_examples": "=",
        },
        {
            "state": "EXPECT_INTEGER",
            "generated": '{"name": "Alice", "age":',
            "allowed": "数字 (0-9)",
            "next_token": " 30",
            "blocked_examples": '"thirty", true, null',
        },
        {
            "state": "OBJECT_END",
            "generated": '{"name": "Alice", "age": 30',
            "allowed": "}",
            "next_token": "}",
            "blocked_examples": ', "extra"',
        },
    ]

    print(f"{'步骤':<4} {'FSM 状态':<18} {'允许的 token':<22} {'已生成'}")
    print("-" * 80)

    for i, step in enumerate(steps):
        print(f"{i+1:<4} {step['state']:<18} {step['allowed']:<22} "
              f"{step['generated'] + step['next_token']}")

    final = '{"name": "Alice", "age": 30}'
    print()
    print(f"最终输出: {final}")
    print(f"JSON 合法: {json.loads(final) is not None}")

    print()
    print("关键机制:")
    print("  1. 每一步, FSM 输出当前合法的 token 集合")
    print("  2. 不合法 token 的 logit 被设为 -inf → softmax 后概率 = 0")
    print("  3. 模型只能从合法 token 中选择 → 输出一定合法")
    print("  4. 计算开销: FSM 查询 < 0.1ms/step, 对推理速度影响 < 5%")

    print()
    print("=" * 65)
    print("从 JSON Schema 到 FSM 的编译链路")
    print("=" * 65)
    print()
    print("  Pydantic Model / JSON Schema")
    print("       │")
    print("       ▼")
    print("  正则表达式 (regex)")
    print('       例: \\{"name":\\s*"[^"]*",\\s*"age":\\s*\\d+\\}')
    print("       │")
    print("       ▼")
    print("  确定性有限自动机 (DFA)")
    print("       状态数通常 < 100, 编译耗时 < 100ms")
    print("       │")
    print("       ▼")
    print("  Token Mask 索引表")
    print("       预计算每个 DFA 状态对应的合法 token ID 集合")
    print("       查表 O(1), 一次性开销")

    print()
    print("框架支持:")
    print("  - outlines: 最早的约束解码库, 支持 JSON/regex/choice")
    print("  - vLLM:     内置 guided decoding (基于 outlines 或 xgrammar)")
    print("  - SGLang:   优化了 batch 内 FSM 共享, JSON 解码性能最好")


def outlines_demo(model_name):
    """用 outlines 做约束解码"""
    try:
        import outlines
        from outlines import models, generate
    except ImportError:
        print("需要安装 outlines: pip install outlines")
        print("  pip install outlines torch transformers")
        return

    try:
        from pydantic import BaseModel
    except ImportError:
        print("需要安装 pydantic: pip install pydantic")
        return

    # 定义输出 schema
    class ToolCall(BaseModel):
        tool: str
        arguments: dict

    class PersonInfo(BaseModel):
        name: str
        age: int
        city: str

    print(f"加载模型: {model_name} ...")
    model = models.transformers(model_name, device="auto")

    # 1. JSON Schema 约束
    print()
    print("[1/3] JSON Schema 约束生成")
    print(f"  Schema: {json.dumps(PersonInfo.model_json_schema(), indent=2)}")

    generator = generate.json(model, PersonInfo)
    result = generator(
        "Extract the person info from this text: "
        "Alice is a 28-year-old engineer living in Shanghai."
    )
    print(f"  输出: {result}")
    print(f"  类型: {type(result).__name__}")
    print(f"  JSON: {result.model_dump_json()}")

    # 2. 正则约束
    print()
    print("[2/3] 正则约束 (IPv4 地址)")
    ip_pattern = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    generator = generate.regex(model, ip_pattern)
    result = generator("The server IP address is: ")
    print(f"  输出: {result}")

    # 3. 选择约束 (分类)
    print()
    print("[3/3] 选择约束 (情感分类)")
    choices = ["positive", "negative", "neutral"]
    generator = generate.choice(model, choices)
    result = generator("The sentiment of '这家餐厅太好吃了' is: ")
    print(f"  输出: {result}")
    print(f"  保证是 {choices} 之一: {result in choices}")


def main():
    parser = argparse.ArgumentParser(description="约束解码演示")
    parser.add_argument("--outlines", action="store_true",
                        help="运行 outlines 实际演示 (需要 GPU)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B",
                        help="outlines 使用的模型")
    args = parser.parse_args()

    if args.outlines:
        outlines_demo(args.model)
    else:
        demo_fsm_principle()


if __name__ == "__main__":
    main()
