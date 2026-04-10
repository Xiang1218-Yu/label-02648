#!/usr/bin/env python
"""数据处理脚本 - 用于批量处理和转换数据"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path

import pandas as pd

from config import get_config
from src.data_preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


def process_data(input_path: Path, output_path: Path, config):
    """处理单个数据文件"""
    logger.info(f"处理数据: {input_path} -> {output_path}")

    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_all_features(df_clean, lags=config.LAG_FEATURES, windows=config.ROLLING_WINDOWS)

    df_features.to_csv(output_path, index=False)
    logger.info(f"数据处理完成, 输出形状: {df_features.shape}")


def batch_process(input_dir: Path, output_dir: Path, config):
    """批量处理目录下的所有CSV文件"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in input_dir.glob("*.csv"):
        output_file = output_dir / f"processed_{csv_file.name}"
        process_data(csv_file, output_file, config)


def main():
    parser = argparse.ArgumentParser(description="数据处理脚本")
    parser.add_argument("--input", "-i", type=str, help="输入CSV文件路径或目录")
    parser.add_argument("--output", "-o", type=str, help="输出路径")
    parser.add_argument("--env", "-e", type=str, default="development", help="运行环境")
    parser.add_argument("--batch", "-b", action="store_true", help="批量处理模式")

    args = parser.parse_args()

    config = get_config(args.env)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.batch:
        input_dir = Path(args.input) if args.input else config.DATA_DIR
        output_dir = Path(args.output) if args.output else config.DATA_DIR / "processed"
        batch_process(input_dir, output_dir, config)
    else:
        input_path = Path(args.input) if args.input else config.DATA_PATH
        output_path = Path(args.output) if args.output else config.DATA_DIR / "processed_data.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        process_data(input_path, output_path, config)

    logger.info("脚本执行完成!")


if __name__ == "__main__":
    main()
