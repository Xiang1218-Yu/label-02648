#!/usr/bin/env python
"""完整流水线执行脚本"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import subprocess
from pathlib import Path

from config import get_config

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent


def run_step(step_name: str, script_name: str, args=None):
    """运行单个步骤"""
    logger.info(f"{'='*60}")
    logger.info(f"执行步骤: {step_name}")
    logger.info(f"{'='*60}")

    cmd = [sys.executable, str(SCRIPT_DIR / script_name)]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"步骤 {step_name} 执行成功!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"步骤 {step_name} 执行失败: {e}")
        return False


def run_full_pipeline(config, skip_eda: bool = False):
    """运行完整流水线"""
    steps = [
        ("数据处理", "data_processing.py"),
        ("模型训练", "train_model.py", ["--all"]),
    ]

    if not skip_eda:
        steps.insert(1, ("EDA分析", "analyze.py", ["--full-report"]))

    for step in steps:
        step_args = step[2] if len(step) > 2 else None
        if not run_step(step[0], step[1], step_args):
            logger.error(f"流水线执行中断!")
            sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("🎉 完整流水线执行成功!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="项目流水线执行脚本")
    parser.add_argument("--env", "-e", type=str, default="development", help="运行环境")
    parser.add_argument("--full", "-f", action="store_true", help="执行完整流水线")
    parser.add_argument("--skip-eda", action="store_true", help="跳过EDA分析步骤")
    parser.add_argument("--step", "-s", type=str, choices=["data", "train", "eval", "deploy"], help="执行指定步骤")

    args = parser.parse_args()

    config = get_config(args.env)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.full:
        run_full_pipeline(config, args.skip_eda)
    elif args.step == "data":
        run_step("数据处理", "data_processing.py")
    elif args.step == "train":
        run_step("模型训练", "train_model.py")
    elif args.step == "eval":
        run_step("模型评估", "evaluate.py")
    else:
        parser.print_help()
        print("\n提示: 使用 --full 执行完整流水线，或 --step 指定单个步骤")


if __name__ == "__main__":
    main()
