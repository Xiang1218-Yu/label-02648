"""
探索性数据分析模块 (EDA - Exploratory Data Analysis)
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


class EDAAnalyzer:
    def __init__(self, figsize=(12, 8), style="seaborn-v0_8-darkgrid"):
        """
        初始化EDA分析器

        参数:
            figsize: 图表大小
            style: 绘图风格
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            # 如果 style 不可用，使用默认样式
            try:
                plt.style.use("seaborn-darkgrid")
            except:
                pass  # 使用默认样式
        self.df = None

    def load_data(self, filepath):
        """加载数据"""
        logger.info(f"加载数据: {filepath}")
        self.df = pd.read_csv(filepath)
        if "datetime" in self.df.columns:
            self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        logger.info(f"数据加载完成，形状: {self.df.shape}")
        return self.df

    def basic_info(self):
        """基本信息统计"""
        logger.info("=" * 60)
        logger.info("数据基本信息")
        logger.info("=" * 60)

        print("\n【数据形状】")
        print(f"行数: {self.df.shape[0]}, 列数: {self.df.shape[1]}")

        print("\n【数据类型】")
        print(self.df.dtypes)

        print("\n【缺失值统计】")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("无缺失值")

        print("\n【数值型特征统计】")
        print(self.df.describe())

        print("\n【前5行数据】")
        print(self.df.head())

        return self.df.describe()

    def plot_time_series(self, target_col="orders_total", save_path=None):
        """绘制时间序列图"""
        logger.info("绘制时间序列图...")

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # 1. 完整时间序列
        axes[0].plot(self.df["datetime"], self.df[target_col], linewidth=1, alpha=0.7, color="steelblue")
        axes[0].set_title("订单量时间序列 - 完整视图", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("时间", fontsize=12)
        axes[0].set_ylabel("订单量", fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # 2. 每日订单量趋势
        daily_orders = self.df.groupby(self.df["datetime"].dt.date)[target_col].sum()
        axes[1].plot(daily_orders.index, daily_orders.values, linewidth=2, marker="o", markersize=4, color="coral")
        axes[1].set_title("每日订单量趋势", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("日期", fontsize=12)
        axes[1].set_ylabel("每日总订单量", fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # 3. 每小时平均订单量
        hourly_avg = self.df.groupby("hour")[target_col].mean()
        axes[2].bar(hourly_avg.index, hourly_avg.values, color="mediumseagreen", alpha=0.7, edgecolor="black")
        axes[2].set_title("每小时平均订单量分布", fontsize=14, fontweight="bold")
        axes[2].set_xlabel("小时", fontsize=12)
        axes[2].set_ylabel("平均订单量", fontsize=12)
        axes[2].set_xticks(range(0, 24))
        axes[2].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"时间序列图已保存: {save_path}")

        plt.close()
        return fig

    def plot_distribution(self, target_col="orders_total", save_path=None):
        """绘制分布图"""
        logger.info("绘制分布图...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 直方图
        axes[0, 0].hist(self.df[target_col], bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        axes[0, 0].set_title("订单量分布直方图", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("订单量", fontsize=12)
        axes[0, 0].set_ylabel("频数", fontsize=12)
        axes[0, 0].grid(True, alpha=0.3, axis="y")

        # 2. 箱线图
        axes[0, 1].boxplot(
            self.df[target_col], vert=True, patch_artist=True, boxprops=dict(facecolor="lightcoral", alpha=0.7)
        )
        axes[0, 1].set_title("订单量箱线图", fontsize=14, fontweight="bold")
        axes[0, 1].set_ylabel("订单量", fontsize=12)
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        # 3. Q-Q图（正态性检验）
        stats.probplot(self.df[target_col], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q图（正态性检验）", fontsize=14, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 核密度估计图
        self.df[target_col].plot(kind="kde", ax=axes[1, 1], linewidth=2, color="purple")
        axes[1, 1].set_title("订单量核密度估计", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("订单量", fontsize=12)
        axes[1, 1].set_ylabel("密度", fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"分布图已保存: {save_path}")

        plt.close()
        return fig

    def plot_correlation_matrix(self, save_path=None):
        """绘制相关性矩阵热力图"""
        logger.info("绘制相关性矩阵...")

        # 选择数值型特征
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 12))

        # 绘制热力图
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title("特征相关性矩阵热力图", fontsize=16, fontweight="bold", pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"相关性矩阵已保存: {save_path}")

        plt.close()
        return fig

    def plot_categorical_analysis(self, save_path=None):
        """分类变量分析"""
        logger.info("绘制分类变量分析图...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 工作日vs周末订单量对比
        if "is_weekend" in self.df.columns:
            weekend_data = self.df.groupby("is_weekend")["orders_total"].agg(["mean", "std"])
            weekend_labels = ["工作日", "周末"]
            axes[0, 0].bar(
                weekend_labels,
                weekend_data["mean"],
                yerr=weekend_data["std"],
                color=["steelblue", "coral"],
                alpha=0.7,
                capsize=5,
                edgecolor="black",
            )
            axes[0, 0].set_title("工作日 vs 周末订单量对比", fontsize=14, fontweight="bold")
            axes[0, 0].set_ylabel("平均订单量", fontsize=12)
            axes[0, 0].grid(True, alpha=0.3, axis="y")

        # 2. 天气影响分析
        if "weather" in self.df.columns:
            weather_data = self.df.groupby("weather")["orders_total"].mean()
            weather_labels = ["晴天", "多云", "小雨", "大雨", "极端天气"]
            axes[0, 1].bar(
                range(len(weather_data)),
                weather_data.values,
                color=sns.color_palette("RdYlBu_r", len(weather_data)),
                alpha=0.7,
                edgecolor="black",
            )
            axes[0, 1].set_xticks(range(len(weather_data)))
            axes[0, 1].set_xticklabels(weather_labels[: len(weather_data)], rotation=45)
            axes[0, 1].set_title("天气对订单量的影响", fontsize=14, fontweight="bold")
            axes[0, 1].set_ylabel("平均订单量", fontsize=12)
            axes[0, 1].grid(True, alpha=0.3, axis="y")

        # 3. 星期几订单量分布
        if "weekday" in self.df.columns:
            weekday_data = self.df.groupby("weekday")["orders_total"].mean()
            weekday_labels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            axes[1, 0].plot(
                range(7), weekday_data.values, marker="o", markersize=10, linewidth=2, color="mediumseagreen"
            )
            axes[1, 0].set_xticks(range(7))
            axes[1, 0].set_xticklabels(weekday_labels)
            axes[1, 0].set_title("星期几订单量趋势", fontsize=14, fontweight="bold")
            axes[1, 0].set_ylabel("平均订单量", fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)

        # 4. 节假日影响分析
        if "is_holiday" in self.df.columns:
            holiday_data = self.df.groupby("is_holiday")["orders_total"].agg(["mean", "std"])
            holiday_labels = ["普通日", "节假日"]
            axes[1, 1].bar(
                holiday_labels,
                holiday_data["mean"],
                yerr=holiday_data["std"],
                color=["lightblue", "gold"],
                alpha=0.7,
                capsize=5,
                edgecolor="black",
            )
            axes[1, 1].set_title("节假日 vs 普通日订单量对比", fontsize=14, fontweight="bold")
            axes[1, 1].set_ylabel("平均订单量", fontsize=12)
            axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"分类变量分析图已保存: {save_path}")

        plt.close()
        return fig

    def plot_region_analysis(self, save_path=None):
        """区域订单分析"""
        logger.info("绘制区域订单分析图...")

        region_cols = [col for col in self.df.columns if col.startswith("region_") and col.endswith("_orders")]

        if not region_cols:
            logger.warning("未找到区域订单数据")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 各区域总订单量对比
        region_totals = self.df[region_cols].sum()
        region_names = [col.replace("region_", "").replace("_orders", "").upper() for col in region_cols]
        axes[0, 0].bar(
            region_names,
            region_totals.values,
            color=sns.color_palette("Set2", len(region_cols)),
            alpha=0.7,
            edgecolor="black",
        )
        axes[0, 0].set_title("各区域总订单量对比", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("区域", fontsize=12)
        axes[0, 0].set_ylabel("总订单量", fontsize=12)
        axes[0, 0].grid(True, alpha=0.3, axis="y")

        # 2. 各区域订单量占比饼图
        axes[0, 1].pie(
            region_totals.values,
            labels=region_names,
            autopct="%1.1f%%",
            colors=sns.color_palette("Set2", len(region_cols)),
            startangle=90,
        )
        axes[0, 1].set_title("各区域订单量占比", fontsize=14, fontweight="bold")

        # 3. 各区域订单量时间序列
        for col, name in zip(region_cols, region_names):
            hourly_avg = self.df.groupby("hour")[col].mean()
            axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker="o", label=f"区域{name}", linewidth=2)
        axes[1, 0].set_title("各区域每小时平均订单量", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("小时", fontsize=12)
        axes[1, 0].set_ylabel("平均订单量", fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 各区域订单量箱线图
        region_data = [self.df[col].values for col in region_cols]
        bp = axes[1, 1].boxplot(region_data, labels=region_names, patch_artist=True)
        for patch, color in zip(bp["boxes"], sns.color_palette("Set2", len(region_cols))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 1].set_title("各区域订单量分布", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("区域", fontsize=12)
        axes[1, 1].set_ylabel("订单量", fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"区域分析图已保存: {save_path}")

        plt.close()
        return fig

    def generate_full_report(self, output_dir="results/eda"):
        """生成完整的EDA报告"""
        logger.info("=" * 60)
        logger.info("开始生成完整EDA报告")
        logger.info("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        # 1. 基本信息
        self.basic_info()

        # 2. 时间序列分析
        self.plot_time_series(save_path=os.path.join(output_dir, "time_series.png"))

        # 3. 分布分析
        self.plot_distribution(save_path=os.path.join(output_dir, "distribution.png"))

        # 4. 相关性分析
        self.plot_correlation_matrix(save_path=os.path.join(output_dir, "correlation_matrix.png"))

        # 5. 分类变量分析
        self.plot_categorical_analysis(save_path=os.path.join(output_dir, "categorical_analysis.png"))

        # 6. 区域分析
        self.plot_region_analysis(save_path=os.path.join(output_dir, "region_analysis.png"))

        logger.info("=" * 60)
        logger.info(f"EDA报告生成完成，保存在: {output_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    # 测试EDA模块
    import sys

    sys.path.append("..")
    from data.data_generator import generate_simulated_data

    logging.basicConfig(level=logging.INFO)
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 生成测试数据
    df = generate_simulated_data(days=30)
    df.to_csv("../data/test_data.csv", index=False)

    # 执行EDA分析
    eda = EDAAnalyzer()
    eda.load_data("../data/test_data.csv")
    eda.generate_full_report()

    print("\nEDA分析完成！")
