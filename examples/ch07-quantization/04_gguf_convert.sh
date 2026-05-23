#!/bin/bash
# GGUF 格式转换脚本
# 用法: bash 04_gguf_convert.sh <model_path> [quantization_type]
# 示例: bash 04_gguf_convert.sh ./Qwen2-7B Q4_K_M

set -e

MODEL_PATH=${1:?"用法: bash 04_gguf_convert.sh <model_path> [quant_type]"}
QUANT_TYPE=${2:-"Q4_K_M"}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-"./llama.cpp"}

echo "========================================"
echo "GGUF 格式转换"
echo "模型路径: $MODEL_PATH"
echo "量化类型: $QUANT_TYPE"
echo "========================================"

# 1. 克隆 llama.cpp（如果不存在）
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "克隆 llama.cpp ..."
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
    cd "$LLAMA_CPP_DIR"
    pip install -r requirements.txt
    make -j$(nproc)
    cd -
fi

# 2. 转换为 GGUF FP16
OUTPUT_DIR="$(dirname $MODEL_PATH)/gguf"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Step 1: 转换为 GGUF FP16 ..."
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MODEL_PATH" \
    --outfile "$OUTPUT_DIR/model-f16.gguf" \
    --outtype f16

echo ""
echo "Step 2: 量化为 $QUANT_TYPE ..."
"$LLAMA_CPP_DIR/llama-quantize" \
    "$OUTPUT_DIR/model-f16.gguf" \
    "$OUTPUT_DIR/model-${QUANT_TYPE}.gguf" \
    "$QUANT_TYPE"

echo ""
echo "常见量化类型说明:"
echo "  Q4_0     - 4-bit，基础量化，速度最快"
echo "  Q4_K_M   - 4-bit，k-quant medium，精度和速度的平衡（推荐）"
echo "  Q5_K_M   - 5-bit，k-quant medium，精度更好"
echo "  Q8_0     - 8-bit，精度最好但文件大"

echo ""
echo "文件大小对比:"
ls -lh "$OUTPUT_DIR/"*.gguf

echo ""
echo "完成! 量化后的模型: $OUTPUT_DIR/model-${QUANT_TYPE}.gguf"
echo ""
echo "用 llama.cpp 测试:"
echo "  $LLAMA_CPP_DIR/llama-cli -m $OUTPUT_DIR/model-${QUANT_TYPE}.gguf -p '你好' -n 100"
echo ""
echo "用 Ollama 导入:"
echo "  创建 Modelfile:"
echo "    FROM $OUTPUT_DIR/model-${QUANT_TYPE}.gguf"
echo "  然后: ollama create my-model -f Modelfile"
