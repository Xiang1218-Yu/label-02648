#!/usr/bin/env python
"""模型评估脚本 - 批量评估多个模型并生成报告"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from config import get_config
from src.model_evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


def load_models(model_dir: Path, model_names: list):
    """从目录加载模型"""
    models = {}
    for model_file in model_dir.glob("*.joblib"):
        model_name = model_file.stem.replace("_model", "")
        if not model_names or model_name in model_names:
            models[model_name] = joblib.load(model_file)
            logger.info(f"已加载模型: {model_name}")
    return models


def run_evaluation(config, model_names: list = None, output_formats: list = None):
    """运行模型评估"""
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

    models = load_models(config.MODELS_DIR, model_names)
    if not models:
        logger.warning("未找到任何已训练的模型，先进行训练...")
        from src.model_training import ModelTrainer

        trainer = ModelTrainer(random_state=config.RANDOM_STATE)
        models = trainer.train_all_models(X_train, y_train)
        trainer.save_models(str(config.MODELS_DIR))

    evaluator = ModelEvaluator(models)
    metrics = evaluator.calculate_metrics(X_test, y_test)

    output_formats = output_formats or ["excel"]
    if "excel" in output_formats:
        metrics_df = evaluator.export_metrics_to_excel(str(config.RESULTS_DIR / "evaluation_metrics.xlsx"))
        print("\n模型性能评估结果:")
        print(metrics_df.to_string())

    if "plots" in output_formats:
        evaluator.plot_predictions(
            X_test, y_test, sample_size=200, save_path=str(config.RESULTS_DIR / "predictions_plot.png")
        )
        evaluator.plot_feature_importance(
            X.columns, top_n=15, save_path=str(config.RESULTS_DIR / "feature_importance.png")
        )

    if "json" in output_formats:
        import json

        with open(config.RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"指标JSON已保存: {config.RESULTS_DIR / 'metrics.json'}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument("--model", "-m", type=str, action="append", help="指定评估的模型(可多次)")
    parser.add_argument("--env", "-e", type=str, default="development", help="运行环境")
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        action="append",
        choices=["excel", "plots", "json"],
        default=["excel", "plots"],
        help="输出格式",
    )

    args = parser.parse_args()
    config = get_config(args.env)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    run_evaluation(config, model_names=args.model, output_formats=args.format)
    logger.info("模型评估脚本执行完成!")
    logger.info(f"结果文件目录: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
