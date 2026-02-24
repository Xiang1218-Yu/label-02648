"""
项目配置文件
"""
import os

class Config:
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # 数据配置
    DATA_PATH = os.path.join(DATA_DIR, 'simulated_data.csv')
    SIMULATION_DAYS = 90
    SAVE_RESULTS = True
    
    # 模型配置
    MODELS_TO_TRAIN = ['random_forest', 'xgboost']
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # EDA配置
    ENABLE_EDA = True  # 是否启用探索性数据分析
    
    # 特征工程配置
    LAG_FEATURES = [1, 2, 3, 6, 12]
    ROLLING_WINDOWS = [3, 6, 12, 24]
    
    # 超参数优化配置
    USE_OPTUNA = False
    N_TRIALS = 100
    
    # 调度优化配置
    RIDER_EFFICIENCY = 10  # 单/小时
    MAX_RIDERS_PER_REGION = 50
    N_HOTSPOT_CLUSTERS = 3
    
    # 可视化配置
    PLOT_STYLE = 'default'  # 使用默认样式以确保兼容性
    FIGURE_SIZE = (12, 8)
    DPI = 100
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
