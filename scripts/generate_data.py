#!/usr/bin/env python
"""数据生成脚本 - 生成不同规模的模拟数据"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging

from config import get_config
from data.data_generator import generate_simulated_data

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="数据生成脚本")
    parser.add_argument("--days", "-d", type=int, default=None, help="模拟天数")
    parser.add_argument("--regions", "-r", type=int, default=5, help="区域数量")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径")
    parser.add_argument("--env", "-e", type=str, default="development", help="运行环境")

    args = parser.parse_args()
    config = get_config(args.env)

    days = args.days if args.days else config.SIMULATION_DAYS
    output_path = args.output if args.output else config.DATA_PATH

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info(f"生成 {days} 天的模拟数据 (区域数: {args.regions})")
    df = generate_simulated_data(days=days, n_regions=args.regions)
    df.to_csv(str(output_path), index=False)
    logger.info(f"数据已保存: {output_path}")
    logger.info(f"数据形状: {df.shape}")


if __name__ == "__main__":
    main()
