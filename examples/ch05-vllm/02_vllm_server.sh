#!/bin/bash
# 第 5 章 示例 2：启动 vLLM OpenAI 兼容服务
#
# 使用方法:
#   chmod +x 02_vllm_server.sh
#   ./02_vllm_server.sh            # 默认配置
#   ./02_vllm_server.sh --large    # 大模型多卡配置
#   ./02_vllm_server.sh --dev      # 开发调试配置

set -e

MODEL=${MODEL:-"Qwen/Qwen2-7B-Instruct"}
PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}

echo "============================================"
echo "vLLM OpenAI Compatible Server"
echo "============================================"

case "${1}" in
    --large)
        # 大模型多卡配置 (72B 模型需要 4x A100)
        echo "配置: 大模型多卡 (TP=4)"
        vllm serve Qwen/Qwen2-72B-Instruct \
            --host ${HOST} \
            --port ${PORT} \
            --tensor-parallel-size 4 \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.92 \
            --max-num-seqs 128 \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --served-model-name qwen2-72b
        ;;

    --dev)
        # 开发调试配置 (小模型, 快速启动)
        echo "配置: 开发调试 (0.5B 模型)"
        vllm serve Qwen/Qwen2-0.5B-Instruct \
            --host ${HOST} \
            --port ${PORT} \
            --max-model-len 2048 \
            --gpu-memory-utilization 0.5 \
            --served-model-name qwen2-dev
        ;;

    --quantized)
        # AWQ 量化模型配置
        echo "配置: AWQ 量化 (7B-AWQ)"
        vllm serve Qwen/Qwen2-7B-Instruct-AWQ \
            --host ${HOST} \
            --port ${PORT} \
            --quantization awq \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.9 \
            --enable-prefix-caching \
            --served-model-name qwen2-7b-awq
        ;;

    *)
        # 默认配置 (7B 模型, 单卡)
        echo "配置: 默认 (${MODEL}, 单卡)"
        echo "端口: ${PORT}"
        echo ""
        echo "启动后测试:"
        echo "  curl http://localhost:${PORT}/v1/models"
        echo "  curl http://localhost:${PORT}/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
        echo ""

        vllm serve ${MODEL} \
            --host ${HOST} \
            --port ${PORT} \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.9 \
            --max-num-seqs 256 \
            --enable-prefix-caching \
            --served-model-name qwen2-7b
        ;;
esac
