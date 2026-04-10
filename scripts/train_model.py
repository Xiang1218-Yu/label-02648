#!/usr/bin/env python
"""模型训练脚本 - 用于单独训练和保存模型"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from config import get_config
from src.model_training import ModelTrainer

logger = logging.getLogger(__name__)


def train_single_model(model_name: str, config, save: bool = True):
    """训练单个模型"""
    logger.info(f"训练模型: {model_name}")

    df = pd.read_csv(str(config.DATA_PATH))
    df["datetime"] = pd.to_datetime(df["datetime"])

    from src.data_preprocessing import DataPreprocessor
    from src.feature_engineering import create_advanced_features, create_region_features

    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_all_features(df_clean, lags=config.LAG_FEATURES, windows=config.ROLLING_WINDOWS)
    df_advanced = create_advanced_features(df_features)
    df_final = create_region_features(df_advanced)

    exclude_cols = ["datetime", "orders_total", "busiest_region"]
    region_order_cols = [col for col in df_final.columns if col.startswith("region_") and col.endswith("_orders")]
    exclude_cols.extend(region_order_cols)
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]

    X = df_final[feature_cols]
    y = df_final["orders_total"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, shuffle=False
    )

    trainer = ModelTrainer(random_state=config.RANDOM_STATE)
    model = trainer.train_single_model(X_train, y_train, model_name=model_name)

    logger.info(f"模型 {model_name} 训练完成!")

    if save:
        save_path = config.MODELS_DIR / f"{model_name}_model.joblib"
        trainer._save_model(model, str(save_path))
        logger.info(f"模型已保存: {save_path}")

    return model, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description="模型训练脚本")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="xgboost",
        choices=["random_forest", "xgboost", "lightgbm"],
        help="要训练的模型类型",
    )
    parser.add_argument("--env", "-e", type=str, default="development", help="运行环境")
    parser.add_argument("--no-save", action="store_true", help="不保存模型")
    parser.add_argument("--all", "-a", action="store_true", help="训练所有配置的模型")

    args = parser.parse_args()

    config = get_config(args.env)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.all:
        for model_name in config.MODELS_TO_TRAIN:
            train_single_model(model_name, config, save=not args.no_save)
    else:
        train_single_model(args.model, config, save=not args.no_save)

    logger.info("模型训练脚本执行完成!")


if __name__ == "__main__":
    main()
