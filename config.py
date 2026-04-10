"""
项目配置文件
支持从 .env 文件加载环境变量
"""
import os
from pathlib import Path


# 尝试加载 python-dotenv，如果未安装则跳过
try:
    from dotenv import load_dotenv
    # 加载 .env 文件
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    """解析逗号分隔的整数列表"""
    if not value:
        return default
    try:
        return [int(x.strip()) for x in value.split(',') if x.strip()]
    except ValueError:
        return default


def _parse_float_tuple(value: str | None, default: tuple[float, float]) -> tuple[float, float]:
    """解析逗号分隔的浮点数元组"""
    if not value:
        return default
    try:
        parts = [x.strip() for x in value.split(',') if x.strip()]
        if len(parts) >= 2:
            return (float(parts[0]), float(parts[1]))
    except ValueError:
        pass
    return default


def _parse_bool(value: str | None, default: bool) -> bool:
    """解析布尔值"""
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def _parse_int(value: str | None, default: int) -> int:
    """解析整数"""
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_float(value: str | None, default: float) -> float:
    """解析浮点数"""
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class Config:
    # 基础路径
    BASE_DIR = Path(__file__).parent.resolve()
    
    # =============================================================================
    # 路径配置（可从环境变量覆盖）
    # =============================================================================
    DATA_DIR = Path(os.getenv('DATA_DIR', BASE_DIR / 'data'))
    MODELS_DIR = Path(os.getenv('MODELS_DIR', BASE_DIR / 'models'))
    RESULTS_DIR = Path(os.getenv('RESULTS_DIR', BASE_DIR / 'results'))
    LOGS_DIR = Path(os.getenv('LOGS_DIR', BASE_DIR / 'logs'))
    
    # 数据文件路径
    DATA_PATH = Path(os.getenv('DATA_PATH', DATA_DIR / 'simulated_data.csv'))
    
    # =============================================================================
    # 项目配置
    # =============================================================================
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = _parse_bool(os.getenv('DEBUG'), default=True)
    
    # =============================================================================
    # 数据配置
    # =============================================================================
    SIMULATION_DAYS = _parse_int(os.getenv('SIMULATION_DAYS'), default=90)
    SAVE_RESULTS = _parse_bool(os.getenv('SAVE_RESULTS'), default=True)
    
    # =============================================================================
    # 模型配置
    # =============================================================================
    MODELS_TO_TRAIN = ['random_forest', 'xgboost']
    TEST_SIZE = _parse_float(os.getenv('TEST_SIZE'), default=0.2)
    RANDOM_STATE = _parse_int(os.getenv('RANDOM_STATE'), default=42)
    CV_FOLDS = _parse_int(os.getenv('CV_FOLDS'), default=5)
    
    # =============================================================================
    # EDA配置
    # =============================================================================
    ENABLE_EDA = _parse_bool(os.getenv('ENABLE_EDA'), default=True)
    
    # =============================================================================
    # 特征工程配置
    # =============================================================================
    LAG_FEATURES = _parse_int_list(os.getenv('LAG_FEATURES'), default=[1, 2, 3, 6, 12])
    ROLLING_WINDOWS = _parse_int_list(os.getenv('ROLLING_WINDOWS'), default=[3, 6, 12, 24])
    
    # =============================================================================
    # 超参数优化配置
    # =============================================================================
    USE_OPTUNA = _parse_bool(os.getenv('USE_OPTUNA'), default=False)
    N_TRIALS = _parse_int(os.getenv('N_TRIALS'), default=100)
    
    # =============================================================================
    # 调度优化配置
    # =============================================================================
    RIDER_EFFICIENCY = _parse_int(os.getenv('RIDER_EFFICIENCY'), default=10)
    MAX_RIDERS_PER_REGION = _parse_int(os.getenv('MAX_RIDERS_PER_REGION'), default=50)
    N_HOTSPOT_CLUSTERS = _parse_int(os.getenv('N_HOTSPOT_CLUSTERS'), default=3)
    
    # =============================================================================
    # 可视化配置
    # =============================================================================
    PLOT_STYLE = os.getenv('PLOT_STYLE', 'default')
    FIGURE_SIZE = _parse_float_tuple(os.getenv('FIGURE_SIZE'), default=(12, 8))
    DPI = _parse_int(os.getenv('PLOT_DPI'), default=100)
    
    # =============================================================================
    # 日志配置
    # =============================================================================
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def to_dict(cls) -> dict:
        """将配置转换为字典"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }
