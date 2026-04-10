"""
Pytest 配置文件
定义测试固件和共享配置
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import Config


@pytest.fixture(scope="session")
def project_root():
    """返回项目根目录路径"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config():
    """返回 Config 类实例"""
    return Config


@pytest.fixture(scope="function")
def sample_dataframe():
    """创建示例 DataFrame 用于测试"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'datetime': dates,
        'orders_total': np.random.randint(50, 200, 100),
        'temperature': np.random.uniform(15, 35, 100),
        'humidity': np.random.uniform(40, 90, 100),
        'is_weekend': np.random.choice([0, 1], 100),
        'weather_code': np.random.choice([0, 1, 2, 3], 100),
    })
    return df


@pytest.fixture(scope="function")
def sample_features_df():
    """创建包含特征的示例 DataFrame"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'datetime': dates,
        'orders_total': np.random.randint(50, 200, 100),
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': np.random.choice([0, 1], 100),
        'temperature': np.random.uniform(15, 35, 100),
        'humidity': np.random.uniform(40, 90, 100),
        'weather_code': np.random.choice([0, 1, 2, 3], 100),
        'orders_total_lag_1': np.random.randint(50, 200, 100),
        'orders_total_lag_2': np.random.randint(50, 200, 100),
        'orders_total_rolling_mean_3': np.random.randint(50, 200, 100),
    })
    return df


@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    """创建临时目录"""
    return tmp_path


@pytest.fixture(autouse=True)
def setup_test_env():
    """每个测试前的环境设置"""
    # 测试前设置
    original_cwd = os.getcwd()
    yield
    # 测试后清理
    os.chdir(original_cwd)
