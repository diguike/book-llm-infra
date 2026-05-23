"""
自建 vs 云 API 成本计算器
帮助决策：该自建推理集群还是用云 API？

运行：
  python 02_cost_calculator.py

输出包含：
  - 自建成本明细
  - 云 API 成本
  - 盈亏平衡点（利用率阈值）
  - 混合架构建议
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPUInstance:
    """GPU 实例配置"""
    name: str                      # 实例类型
    gpu_model: str                 # GPU 型号
    gpu_count: int                 # GPU 数量
    gpu_memory_gb: int             # 单卡显存 (GB)
    monthly_cost: float            # 月租金 (元)
    system_disk_cost: float = 140  # 系统盘月费
    data_disk_cost: float = 350    # 数据盘月费
    bandwidth_cost: float = 400    # 带宽月费


@dataclass
class ModelSpec:
    """模型规格"""
    name: str
    params_b: float                # 参数量 (十亿)
    memory_gb: float               # 加载所需显存 (GB, FP16)
    throughput_tps: float          # 系统吞吐 (tokens/s, 并发 32)


@dataclass
class CloudAPIPrice:
    """云 API 定价"""
    provider: str
    model: str
    input_per_mtok: float          # 输入 每百万 token 价格 (元)
    output_per_mtok: float         # 输出 每百万 token 价格 (元)


# ============================================================
# 预设配置
# ============================================================

# 阿里云 GPU 实例
GPU_INSTANCES = {
    "a10": GPUInstance(
        name="ecs.gn7i-c8g1.2xlarge",
        gpu_model="A10",
        gpu_count=1,
        gpu_memory_gb=24,
        monthly_cost=5800,
    ),
    "a100-40g": GPUInstance(
        name="ecs.gn7-c12g1.3xlarge",
        gpu_model="A100-40GB",
        gpu_count=1,
        gpu_memory_gb=40,
        monthly_cost=12000,
    ),
    "a100-80g": GPUInstance(
        name="ecs.gn7e-c16g1.4xlarge",
        gpu_model="A100-80GB",
        gpu_count=1,
        gpu_memory_gb=80,
        monthly_cost=18000,
    ),
}

# 常见模型规格
MODELS = {
    "qwen-7b": ModelSpec(
        name="Qwen2.5-7B-Instruct",
        params_b=7,
        memory_gb=14,
        throughput_tps=1500,
    ),
    "qwen-14b": ModelSpec(
        name="Qwen2.5-14B-Instruct",
        params_b=14,
        memory_gb=28,
        throughput_tps=800,
    ),
    "qwen-72b": ModelSpec(
        name="Qwen2.5-72B-Instruct",
        params_b=72,
        memory_gb=144,
        throughput_tps=400,
    ),
}

# 云 API 定价 (2025 年初)
CLOUD_APIS = [
    CloudAPIPrice("阿里通义", "qwen-plus (7B级)", 0.8, 2.0),
    CloudAPIPrice("阿里通义", "qwen-max (72B级)", 2.0, 6.0),
    CloudAPIPrice("SiliconFlow", "Qwen2.5-7B", 0.35, 0.35),
    CloudAPIPrice("SiliconFlow", "Qwen2.5-72B", 4.13, 4.13),
    CloudAPIPrice("DeepSeek", "deepseek-chat", 1.0, 2.0),
]


# ============================================================
# 成本计算
# ============================================================

def calculate_self_hosted_cost(
    gpu: GPUInstance,
    model: ModelSpec,
    gpu_count: int,
    utilization: float = 0.7,
    ops_salary_monthly: float = 5000,  # 分摊到单实例的运维人力
) -> dict:
    """
    计算自建推理服务的月度成本。

    Args:
        gpu: GPU 实例配置
        model: 模型规格
        gpu_count: 所需 GPU 实例数量
        utilization: 平均利用率 (0-1)
        ops_salary_monthly: 分摊到每个模型实例的运维人力成本

    Returns:
        成本明细
    """
    # 硬件成本
    hw_cost = (gpu.monthly_cost + gpu.system_disk_cost +
               gpu.data_disk_cost + gpu.bandwidth_cost) * gpu_count

    # 总成本
    total_monthly = hw_cost + ops_salary_monthly

    # 吞吐量计算
    monthly_seconds = 30 * 24 * 3600
    monthly_tokens = model.throughput_tps * utilization * monthly_seconds
    cost_per_mtok = total_monthly / (monthly_tokens / 1_000_000)

    return {
        "model": model.name,
        "gpu": f"{gpu.gpu_model} x {gpu_count}",
        "hardware_cost": hw_cost,
        "ops_cost": ops_salary_monthly,
        "total_monthly": total_monthly,
        "utilization": f"{utilization:.0%}",
        "monthly_tokens_m": round(monthly_tokens / 1_000_000, 1),
        "cost_per_mtok": round(cost_per_mtok, 3),
    }


def calculate_cloud_cost(
    api: CloudAPIPrice,
    monthly_tokens_m: float,
    input_ratio: float = 0.75,
) -> dict:
    """
    计算云 API 月度成本。

    Args:
        api: 云 API 定价
        monthly_tokens_m: 月度 token 消耗 (百万)
        input_ratio: 输入 token 占比

    Returns:
        成本明细
    """
    input_tokens_m = monthly_tokens_m * input_ratio
    output_tokens_m = monthly_tokens_m * (1 - input_ratio)

    input_cost = input_tokens_m * api.input_per_mtok
    output_cost = output_tokens_m * api.output_per_mtok
    total = input_cost + output_cost

    blended_per_mtok = total / monthly_tokens_m if monthly_tokens_m > 0 else 0

    return {
        "provider": api.provider,
        "model": api.model,
        "monthly_tokens_m": monthly_tokens_m,
        "input_cost": round(input_cost, 2),
        "output_cost": round(output_cost, 2),
        "total_monthly": round(total, 2),
        "blended_per_mtok": round(blended_per_mtok, 3),
    }


def break_even_analysis(
    self_hosted_monthly: float,
    self_hosted_max_tokens_m: float,
    cloud_per_mtok: float,
) -> dict:
    """
    计算盈亏平衡点。

    在多少 token 消耗量下，自建和云 API 成本相等？
    """
    # 盈亏平衡点：self_hosted_monthly = cloud_per_mtok * X
    break_even_tokens_m = self_hosted_monthly / cloud_per_mtok

    # 对应的利用率
    break_even_util = break_even_tokens_m / self_hosted_max_tokens_m

    return {
        "break_even_tokens_m": round(break_even_tokens_m, 1),
        "break_even_utilization": f"{break_even_util:.1%}",
        "self_hosted_monthly_fixed": self_hosted_monthly,
        "cloud_per_mtok": cloud_per_mtok,
        "recommendation": (
            "自建更划算" if break_even_util < 0.5
            else "云 API 更划算（除非利用率能保证 > 50%）"
        ),
    }


# ============================================================
# 混合架构成本模拟
# ============================================================

def hybrid_cost_simulation(
    self_hosted_monthly: float,
    self_hosted_max_tps: float,
    cloud_per_mtok: float,
    traffic_pattern: list[float],  # 每小时的请求 TPS
) -> dict:
    """
    混合架构成本模拟：基线流量自建，峰值溢出到云 API。

    Args:
        self_hosted_monthly: 自建月度固定成本
        self_hosted_max_tps: 自建最大吞吐 (tokens/s)
        cloud_per_mtok: 云 API 每百万 token 价格
        traffic_pattern: 24 小时的流量模式 (每小时平均 TPS)

    Returns:
        混合架构的月度成本
    """
    self_hosted_tokens = 0
    cloud_tokens = 0

    for hour_tps in traffic_pattern:
        if hour_tps <= self_hosted_max_tps:
            self_hosted_tokens += hour_tps * 3600
        else:
            self_hosted_tokens += self_hosted_max_tps * 3600
            overflow_tps = hour_tps - self_hosted_max_tps
            cloud_tokens += overflow_tps * 3600

    # 月度（30 天）
    self_hosted_tokens_monthly = self_hosted_tokens * 30
    cloud_tokens_monthly = cloud_tokens * 30

    cloud_cost = (cloud_tokens_monthly / 1_000_000) * cloud_per_mtok
    total_cost = self_hosted_monthly + cloud_cost

    # 对比纯云 API
    total_tokens_monthly = (self_hosted_tokens_monthly + cloud_tokens_monthly)
    pure_cloud_cost = (total_tokens_monthly / 1_000_000) * cloud_per_mtok

    return {
        "self_hosted_tokens_m": round(self_hosted_tokens_monthly / 1_000_000, 1),
        "cloud_overflow_tokens_m": round(cloud_tokens_monthly / 1_000_000, 1),
        "self_hosted_fixed_cost": self_hosted_monthly,
        "cloud_overflow_cost": round(cloud_cost, 2),
        "hybrid_total_monthly": round(total_cost, 2),
        "pure_cloud_monthly": round(pure_cloud_cost, 2),
        "savings_vs_pure_cloud": f"{(1 - total_cost / pure_cloud_cost) * 100:.1f}%" if pure_cloud_cost > 0 else "N/A",
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("LLM 推理成本计算器")
    print("=" * 70)

    # --- 场景 1：Qwen-7B on A10 ---
    print("\n📊 场景 1：Qwen2.5-7B on A10")
    print("-" * 50)

    self_hosted = calculate_self_hosted_cost(
        gpu=GPU_INSTANCES["a10"],
        model=MODELS["qwen-7b"],
        gpu_count=1,
        utilization=0.7,
    )
    for k, v in self_hosted.items():
        print(f"  {k}: {v}")

    # 对比 SiliconFlow
    cloud = calculate_cloud_cost(
        api=CLOUD_APIS[2],  # SiliconFlow 7B
        monthly_tokens_m=self_hosted["monthly_tokens_m"],
    )
    print(f"\n  云 API 对比 (SiliconFlow):")
    for k, v in cloud.items():
        print(f"    {k}: {v}")

    # 盈亏平衡
    bep = break_even_analysis(
        self_hosted_monthly=self_hosted["total_monthly"],
        self_hosted_max_tokens_m=MODELS["qwen-7b"].throughput_tps * 30 * 24 * 3600 / 1_000_000,
        cloud_per_mtok=cloud["blended_per_mtok"],
    )
    print(f"\n  盈亏平衡分析:")
    for k, v in bep.items():
        print(f"    {k}: {v}")

    # --- 场景 2：Qwen-72B on A100-80G x2 ---
    print("\n\n📊 场景 2：Qwen2.5-72B on A100-80GB x2")
    print("-" * 50)

    self_hosted_72b = calculate_self_hosted_cost(
        gpu=GPU_INSTANCES["a100-80g"],
        model=MODELS["qwen-72b"],
        gpu_count=2,
        utilization=0.6,
    )
    for k, v in self_hosted_72b.items():
        print(f"  {k}: {v}")

    cloud_72b = calculate_cloud_cost(
        api=CLOUD_APIS[3],  # SiliconFlow 72B
        monthly_tokens_m=self_hosted_72b["monthly_tokens_m"],
    )
    print(f"\n  云 API 对比 (SiliconFlow 72B):")
    for k, v in cloud_72b.items():
        print(f"    {k}: {v}")

    # --- 场景 3：混合架构模拟 ---
    print("\n\n📊 场景 3：混合架构（7B, A10 x 2 + 云 API 溢出）")
    print("-" * 50)

    # 典型的日流量模式：白天高峰，夜间低谷
    # 单位：tokens/s
    daily_traffic = [
        200,  200,  150,  100,  100,  150,   # 0-5 点：低谷
        300,  600,  900, 1200, 1400, 1500,   # 6-11 点：上午高峰
        1300, 1400, 1500, 1600, 1400, 1200,  # 12-17 点：下午高峰
        1000,  800,  600,  500,  400,  300,  # 18-23 点：晚间回落
    ]

    hybrid = hybrid_cost_simulation(
        self_hosted_monthly=calculate_self_hosted_cost(
            GPU_INSTANCES["a10"], MODELS["qwen-7b"], 2, 0.95
        )["total_monthly"],
        self_hosted_max_tps=1500 * 2 * 0.95,  # 2 台 A10，95% 利用率上限
        cloud_per_mtok=0.35,
        traffic_pattern=daily_traffic,
    )
    for k, v in hybrid.items():
        print(f"  {k}: {v}")

    # --- 汇总建议 ---
    print("\n\n" + "=" * 70)
    print("决策建议")
    print("=" * 70)
    print("""
  1. 小模型 (7B) + 低成本推理平台 (SiliconFlow)
     → 云 API 通常更划算，除非日均利用率 > 70%

  2. 大模型 (72B) + 自建
     → 利用率 > 40% 时自建划算（云 API 大模型定价偏高）

  3. 混合架构
     → 最佳实践：2 台自建扛基线，峰值溢出到云 API
     → 比纯云 API 节省 20-40%

  4. 额外考虑因素
     → 数据安全：金融/医疗必须自建
     → 延迟要求：自建延迟更稳定
     → 微调需求：自建可以跑自定义模型
     → 运维能力：团队没有 GPU 运维经验建议先用云 API
""")


if __name__ == "__main__":
    main()
