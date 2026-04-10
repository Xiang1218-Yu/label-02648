"""
外卖订单预测与调度优化系统 - 核心模块包
"""

__version__ = "1.1.0"
__author__ = "Data Analysis Team"

from .data_preprocessing import DataPreprocessor
from .eda import EDAAnalyzer
from .feature_engineering import create_advanced_features, create_region_features
from .model_evaluation import ModelEvaluator
from .model_training import ModelTrainer
from .scheduling_optimization import SchedulingOptimizer
from .utils import convert_to_json_serializable, safe_float, safe_int

__all__ = [
    "DataPreprocessor",
    "create_advanced_features",
    "create_region_features",
    "ModelTrainer",
    "ModelEvaluator",
    "SchedulingOptimizer",
    "EDAAnalyzer",
    "convert_to_json_serializable",
    "safe_int",
    "safe_float",
]
