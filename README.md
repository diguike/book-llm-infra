# LLM Infra 从入门到实践

> 在线阅读 · [inferloop.dev/llm-infra](https://inferloop.dev/llm-infra)  
> 所有书目 · [inferloop.dev](https://inferloop.dev)

> 一本写给 Agent 开发者的 LLM 底层技术指南

面向应用层工程师（前端/全栈/Agent 开发者），系统化学习 LLM 基础设施知识。

## 目标

- 读完后对 LLM Infra 有完整认知，具备初步实战能力
- 每章配套可运行的代码示例
- 所有部署示例基于阿里云/腾讯云，确保国内可用

## 目录

- [前言](./chapters/preface/README.md)
- [第 0 章　Python 环境与快速入门](./chapters/ch00-python-quickstart/README.md)

**第一部分　基础认知重建**

- [第 1 章　全景图：LLM 技术栈的分层](./chapters/ch01-overview/README.md)
- [第 2 章　Transformer 架构：工程师视角](./chapters/ch02-transformer/README.md)
- [第 3 章　GPU 与计算基础](./chapters/ch03-gpu-basics/README.md)

**第二部分　模型服务与推理引擎**

- [第 4 章　模型推理：从权重文件到 API 响应](./chapters/ch04-inference-basics/README.md)
- [第 5 章　vLLM：工业级推理引擎深度剖析](./chapters/ch05-vllm/README.md)
- [第 6 章　推理引擎对比与选型](./chapters/ch06-engines-comparison/README.md)

**第三部分　模型优化**

- [第 7 章　量化：用更少的显存跑更大的模型](./chapters/ch07-quantization/README.md)
- [第 8 章　推理加速技术](./chapters/ch08-inference-optimization/README.md)

**第四部分　微调与训练**

- [第 9 章　微调：让通用模型变成领域专家](./chapters/ch09-finetuning/README.md)
- [第 10 章　RLHF 与对齐](./chapters/ch10-alignment/README.md)
- [第 11 章　分布式训练基础](./chapters/ch11-distributed-training/README.md)

**第五部分　生产部署与平台工程**

- [第 12 章　LLM 服务的生产化部署](./chapters/ch12-production-deploy/README.md)
- [第 13 章　可观测性与成本优化](./chapters/ch13-observability-cost/README.md)
- [第 14 章　RAG 系统的基础设施](./chapters/ch14-rag-infra/README.md)

**第六部分　前沿与进阶**

- [第 15 章　多模态模型的基础设施](./chapters/ch15-multimodal/README.md)
- [第 16 章　延伸学习地图](./chapters/ch16-career-path/README.md)

**附录**

- [附录 A　阿里云/腾讯云 GPU 实例速查](./appendix/cloud-gpu-guide.md)
- [附录 C　术语表](./appendix/glossary.md)

> 附录 B（Python/PyTorch 环境搭建）已并入第 0 章。

## 快速开始

```bash
# 克隆项目
git clone https://github.com/diguike/book-llm-infra.git
cd book-llm-infra

# 安装 Python 依赖（推荐 Python 3.10+）
pip install -r requirements.txt

# 进入某一章的示例
cd examples/ch02-transformer
```

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- 部分章节需要 GPU（最低 RTX 3090 / [阿里云](https://www.aliyun.com/minisite/goods?userCode=okjhlpr5) ecs.gn7i / [腾讯云](https://cloud.tencent.com/act/pro/featured-202607?from=30156&Is=sdk-topnav&cps_key=1d358d18a7a17b4a6df8d67a62fd3d3d) GN10Xp）
- 各章节的 README 会标注具体硬件要求

## 在线阅读

[飞书 Wiki](https://fivwvysqdz.feishu.cn/wiki/MYBqwDqvpiJFF4kFj7xc2S78nug)

## License

CC BY-NC-SA 4.0


## 相关书

来自同一作者的其他书:

- [《Hermes Agent 源码解读》](https://inferloop.dev/hermes-agent)
- [《AI Token 中转站实战》](https://inferloop.dev/llm-gateway)
- [《Agent Memory 工程实战》](https://inferloop.dev/claude-mem)
- [《百万级 AI Agent 平台架构》](https://inferloop.dev/enterprise-agent)
- [《OpenClaw 源码解析》](https://inferloop.dev/openclaw)
- [《Transformer 教学》](https://inferloop.dev/transformer)
- [《Claude Code Skill 开发指南》](https://inferloop.dev/claude-skill)
- [《Claude 插件官方指南》](https://inferloop.dev/claude-plugins)
- [《自己动手写 AI Agent》](https://inferloop.dev/ling-agent)
