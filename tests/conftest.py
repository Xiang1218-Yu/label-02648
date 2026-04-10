"""Pytest 配置和固件"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_config():
    """测试环境配置"""
    from config import get_config

    return get_config("test")


@pytest.fixture(scope="session")
def sample_data():
    """生成测试用的模拟数据"""
    dates = pd.date_range(start="2024-01-01", periods=24 * 7, freq="h")
    df = pd.DataFrame(
        {
            "datetime": dates,
            "orders_total": np.random.randint(10, 100, size=len(dates)),
            "temperature": np.random.uniform(10, 35, size=len(dates)),
            "weather": np.random.choice([0, 1, 2], size=len(dates)),
            "region_a_orders": np.random.randint(0, 30, size=len(dates)),
            "region_b_orders": np.random.randint(0, 30, size=len(dates)),
        }
    )
    return df


@pytest.fixture(scope="session")
def sample_features():
    """简单特征和标签用于测试"""
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.rand(100) * 100
    return X, y
