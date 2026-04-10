"""
项目配置文件 - 支持多环境配置和环境变量覆盖
"""

import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"


class BaseConfig(BaseSettings):
    """基础配置类 - 使用pydantic-settings支持环境变量"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT, description="运行环境")

    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.resolve(), description="项目根目录")
    DATA_DIR: Optional[Path] = None
    MODELS_DIR: Optional[Path] = None
    RESULTS_DIR: Optional[Path] = None
    LOGS_DIR: Optional[Path] = None
    CONFIG_DIR: Optional[Path] = None

    DATA_PATH: Optional[Path] = None
    SIMULATION_DAYS: int = Field(default=90, ge=1, description="模拟数据天数")
    SAVE_RESULTS: bool = Field(default=True, description="是否保存结果")

    MODELS_TO_TRAIN: List[str] = Field(default_factory=lambda: ["random_forest", "xgboost"])
    TEST_SIZE: float = Field(default=0.2, ge=0.0, le=1.0)
    RANDOM_STATE: int = Field(default=42, ge=0)
    CV_FOLDS: int = Field(default=5, ge=2, le=10)

    ENABLE_EDA: bool = Field(default=True, description="是否启用探索性数据分析")

    LAG_FEATURES: List[int] = Field(default_factory=lambda: [1, 2, 3, 6, 12])
    ROLLING_WINDOWS: List[int] = Field(default_factory=lambda: [3, 6, 12, 24])

    USE_OPTUNA: bool = Field(default=False)
    N_TRIALS: int = Field(default=100, ge=10)

    RIDER_EFFICIENCY: int = Field(default=10, ge=1, description="骑手效率: 单/小时")
    MAX_RIDERS_PER_REGION: int = Field(default=50, ge=1)
    N_HOTSPOT_CLUSTERS: int = Field(default=3, ge=1)

    PLOT_STYLE: str = Field(default="default")
    FIGURE_SIZE: Tuple[int, int] = Field(default=(12, 8))
    DPI: int = Field(default=100, ge=50, le=300)

    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def model_post_init(self, __context) -> None:
        """初始化后动态设置路径"""
        if self.DATA_DIR is None:
            self.DATA_DIR = self.BASE_DIR / "data"
        if self.MODELS_DIR is None:
            self.MODELS_DIR = self.BASE_DIR / "models"
        if self.RESULTS_DIR is None:
            self.RESULTS_DIR = self.BASE_DIR / "results"
        if self.LOGS_DIR is None:
            self.LOGS_DIR = self.BASE_DIR / "logs"
        if self.CONFIG_DIR is None:
            self.CONFIG_DIR = self.BASE_DIR / "configs"
        if self.DATA_PATH is None:
            self.DATA_PATH = self.DATA_DIR / "simulated_data.csv"
        self.create_directories()

    def create_directories(self) -> None:
        """创建必要的目录"""
        for directory in [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR, self.LOGS_DIR, self.CONFIG_DIR]:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(BaseConfig):
    """开发环境配置"""

    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT)
    LOG_LEVEL: str = Field(default="DEBUG")
    USE_OPTUNA: bool = Field(default=False)


class ProductionConfig(BaseConfig):
    """生产环境配置"""

    ENVIRONMENT: Environment = Field(default=Environment.PRODUCTION)
    LOG_LEVEL: str = Field(default="INFO")
    USE_OPTUNA: bool = Field(default=True)
    N_TRIALS: int = Field(default=200, ge=10)


class TestConfig(BaseConfig):
    """测试环境配置"""

    ENVIRONMENT: Environment = Field(default=Environment.TEST)
    SIMULATION_DAYS: int = Field(default=7, ge=1)
    USE_OPTUNA: bool = Field(default=False)
    SAVE_RESULTS: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="DEBUG")


def get_config(env: Optional[str] = None) -> BaseConfig:
    """
    根据环境变量或指定环境获取配置
    Usage:
        config = get_config()  # 自动从环境变量获取
        config = get_config("production")  # 指定生产环境
    """
    if env is None:
        env = os.getenv("APP_ENV", "development").lower()

    config_map = {
        "development": DevelopmentConfig,
        "dev": DevelopmentConfig,
        "production": ProductionConfig,
        "prod": ProductionConfig,
        "test": TestConfig,
    }

    config_class = config_map.get(env, DevelopmentConfig)
    return config_class()


Config = get_config()
