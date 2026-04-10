"""
数据生成模块 - 生成模拟外卖订单数据
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_simulated_data(days=90, seed=42):
    """
    生成模拟外卖订单数据

    参数:
        days: 模拟天数，默认90天
        seed: 随机种子

    返回:
        pandas DataFrame，包含时间特征、环境特征和订单量
    """
    np.random.seed(seed)

    # 1. 创建时间序列（按小时）
    start_date = datetime(2024, 1, 1)
    hours = days * 24
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]

    df = pd.DataFrame({"datetime": timestamps})
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["day"] = df["datetime"].dt.day
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # 2. 生成节假日（简化：每月1号和15号）
    df["is_holiday"] = ((df["day"] == 1) | (df["day"] == 15)).astype(int)

    # 3. 生成环境特征
    # 温度：15-35度，带季节性变化
    df["temperature"] = 25 + 10 * np.sin(2 * np.pi * np.arange(hours) / (24 * 30)) + np.random.normal(0, 3, hours)
    df["temperature"] = df["temperature"].clip(15, 35)

    # 天气：0=晴天, 1=多云, 2=小雨, 3=大雨, 4=极端天气
    weather_probs = [0.5, 0.3, 0.15, 0.04, 0.01]
    df["weather"] = np.random.choice([0, 1, 2, 3, 4], size=hours, p=weather_probs)

    # 4. 生成订单量（核心算法）
    base_orders = 100  # 基础订单量

    # 每日双峰模式（午餐11-13点，晚餐18-20点）
    hour_pattern = np.array(
        [
            0.3,
            0.2,
            0.1,
            0.1,
            0.1,
            0.2,  # 0-5点：深夜低谷
            0.4,
            0.6,
            0.8,
            1.0,
            1.2,
            1.8,  # 6-11点：早餐+午餐前
            2.0,
            1.8,
            1.2,
            1.0,
            0.9,
            1.0,  # 12-17点：午餐高峰+下午
            1.5,
            2.2,
            2.0,
            1.5,
            1.0,
            0.6,  # 18-23点：晚餐高峰+夜宵
        ]
    )

    # 应用时间模式
    orders = base_orders * hour_pattern[df["hour"].values]

    # 周末效应（+20%）
    orders *= 1 + 0.2 * df["is_weekend"].values

    # 节假日效应（+30%）
    orders *= 1 + 0.3 * df["is_holiday"].values

    # 天气影响
    weather_impact = {0: 1.0, 1: 0.95, 2: 0.85, 3: 0.7, 4: 0.5}
    orders *= df["weather"].map(weather_impact).values

    # 温度影响（极端温度减少订单）
    temp_impact = 1 - 0.01 * np.abs(df["temperature"].values - 25)
    orders *= temp_impact

    # 添加泊松噪声
    orders = np.random.poisson(orders)
    df["orders_total"] = orders.astype(int)

    # 5. 生成各区域订单分布（使用狄利克雷分布）
    n_regions = 5
    region_names = ["region_A_orders", "region_B_orders", "region_C_orders", "region_D_orders", "region_E_orders"]

    # 区域权重（A区最繁华，E区最偏远）
    alpha = np.array([3.0, 2.5, 2.0, 1.5, 1.0])

    for i in range(hours):
        # 每小时重新采样区域分布
        region_dist = np.random.dirichlet(alpha)
        total = df.loc[i, "orders_total"]

        # 分配订单到各区域
        region_orders = np.random.multinomial(total, region_dist)
        for j, region_name in enumerate(region_names):
            df.loc[i, region_name] = region_orders[j]

    # 确保区域订单为整数
    for region_name in region_names:
        df[region_name] = df[region_name].astype(int)

    return df


if __name__ == "__main__":
    # 测试数据生成
    df = generate_simulated_data(days=7)
    print("数据生成成功！")
    print(f"数据形状: {df.shape}")
    print(f"\n前5行数据:\n{df.head()}")
    print(f"\n数据统计:\n{df.describe()}")
