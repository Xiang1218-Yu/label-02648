"""项目配置文件（支持分环境配置和环境变量覆盖）。"""

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_DIR = os.path.join(BASE_DIR, "config")
    ENV = os.getenv("APP_ENV", "dev")

    @classmethod
    def _read_json(cls, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def _load_settings(cls) -> Dict[str, Any]:
        base_file = os.path.join(cls.CONFIG_DIR, "settings.base.json")
        env_file = os.path.join(cls.CONFIG_DIR, "settings.{}.json".format(cls.ENV))
        base_cfg = cls._read_json(base_file)
        env_cfg = cls._read_json(env_file)
        return _deep_merge(base_cfg, env_cfg)

    @classmethod
    def _resolve_path(cls, path_value: str) -> str:
        return path_value if os.path.isabs(path_value) else os.path.join(cls.BASE_DIR, path_value)


_settings = Config._load_settings()

_data = _settings.get("data", {})
_model = _settings.get("model", {})
_eda = _settings.get("eda", {})
_feature = _settings.get("feature", {})
_optimizer = _settings.get("optimizer", {})
_paths = _settings.get("paths", {})
_logging = _settings.get("logging", {})

Config.DATA_DIR = Config._resolve_path(_data.get("data_dir", "data"))
Config.MODELS_DIR = Config._resolve_path(_paths.get("models_dir", "models"))
Config.RESULTS_DIR = Config._resolve_path(_paths.get("results_dir", "results"))
Config.LOGS_DIR = Config._resolve_path(_paths.get("logs_dir", "logs"))
Config.DATA_PATH = Config._resolve_path(_data.get("data_path", "data/simulated_data.csv"))

Config.SIMULATION_DAYS = int(os.getenv("SIMULATION_DAYS", _data.get("simulation_days", 90)))
Config.SAVE_RESULTS = str(os.getenv("SAVE_RESULTS", _data.get("save_results", True))).lower() in (
    "1",
    "true",
    "yes",
)

Config.MODELS_TO_TRAIN = _model.get("models_to_train", ["random_forest", "xgboost"])
Config.TEST_SIZE = float(os.getenv("TEST_SIZE", _model.get("test_size", 0.2)))
Config.RANDOM_STATE = int(os.getenv("RANDOM_STATE", _model.get("random_state", 42)))
Config.CV_FOLDS = int(_model.get("cv_folds", 5))

Config.ENABLE_EDA = str(os.getenv("ENABLE_EDA", _eda.get("enable_eda", True))).lower() in (
    "1",
    "true",
    "yes",
)
Config.LAG_FEATURES = _feature.get("lag_features", [1, 2, 3, 6, 12])
Config.ROLLING_WINDOWS = _feature.get("rolling_windows", [3, 6, 12, 24])

Config.USE_OPTUNA = False
Config.N_TRIALS = 100

Config.RIDER_EFFICIENCY = int(_optimizer.get("rider_efficiency", 10))
Config.MAX_RIDERS_PER_REGION = int(_optimizer.get("max_riders_per_region", 50))
Config.N_HOTSPOT_CLUSTERS = int(_optimizer.get("n_hotspot_clusters", 3))

Config.PLOT_STYLE = "default"
Config.FIGURE_SIZE = (12, 8)
Config.DPI = 100

Config.LOG_LEVEL = os.getenv("LOG_LEVEL", _logging.get("level", "INFO"))
Config.LOG_FORMAT = _logging.get(
    "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _create_directories(cls) -> None:
    """创建必要目录。"""
    for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)


Config.create_directories = classmethod(_create_directories)
