"""
合并 LoRA 权重到基础模型

训练完成后，LoRA adapter 是单独保存的（几十 MB）。
要用 vLLM 等推理框架部署，需要先把 adapter 合并回基础模型。

使用方法:
    python 03_merge_lora.py \
        --base_model Qwen/Qwen2-7B \
        --adapter_path ./output/qwen2-7b-qlora \
        --output_path ./output/qwen2-7b-merged

合并后的模型可以直接用 vLLM 部署:
    vllm serve ./output/qwen2-7b-merged --port 8000
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora(base_model_path: str, adapter_path: str, output_path: str):
    """
    把 LoRA adapter 合并回基础模型，保存为标准的 HuggingFace 格式。

    注意：合并时需要加载完整精度的基础模型（bf16），所以需要 ~15GB 显存。
    如果显存不够，可以用 device_map="cpu" 在 CPU 上合并（慢但不需要 GPU）。
    """
    print(f"Loading base model: {base_model_path}")
    # 加载基础模型（bf16 精度，不做量化）
    # 合并时必须用非量化模型，因为量化模型的权重格式不同
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 改成 "cpu" 可以在 CPU 上合并
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging LoRA weights into base model...")
    # 合并权重：W' = W + BA
    # 合并后模型和原始模型结构完全一样，推理时没有额外开销
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)

    # 同时保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("Done!")
    print(f"\nYou can now deploy with vLLM:")
    print(f"  vllm serve {output_path} --port 8000")
    print(f"\nOr test locally:")
    print(f"  from transformers import pipeline")
    print(f'  pipe = pipeline("text-generation", model="{output_path}")')
    print(f'  pipe("Hello, ")')


def test_merged_model(model_path: str, prompt: str = "你好，请介绍一下你自己"):
    """简单测试合并后的模型"""
    print(f"\nTesting merged model with prompt: {prompt}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Response: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen2-7B",
        help="Base model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--adapter_path", type=str, default="./output/qwen2-7b-qlora",
        help="Path to the LoRA adapter directory"
    )
    parser.add_argument(
        "--output_path", type=str, default="./output/qwen2-7b-merged",
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test the merged model after merging"
    )
    args = parser.parse_args()

    merge_lora(args.base_model, args.adapter_path, args.output_path)

    if args.test:
        test_merged_model(args.output_path)
