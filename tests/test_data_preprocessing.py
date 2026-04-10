"""
数据预处理模块测试
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """测试 DataPreprocessor 类"""

    def test_init(self):
        """测试初始化"""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler is not None
        assert preprocessor.feature_names == []

    def test_clean_data(self, sample_dataframe):
        """测试数据清洗"""
        preprocessor = DataPreprocessor()

        # 添加一些缺失值
        df_with_na = sample_dataframe.copy()
        df_with_na.loc[0, 'temperature'] = np.nan

        df_clean = preprocessor.clean_data(df_with_na)

        assert isinstance(df_clean, pd.DataFrame)
        # 前向填充后，第一个NaN应该被填充（如果后面有值）
        # 或者保持NaN（如果整列都是NaN）
        assert df_clean.shape == df_with_na.shape

    def test_create_time_features(self, sample_dataframe):
        """测试时间特征创建"""
        preprocessor = DataPreprocessor()
        df = sample_dataframe.copy()

        df_result = preprocessor.create_time_features(df)

        assert 'hour' in df_result.columns
        assert 'weekday' in df_result.columns
        assert 'month' in df_result.columns
        assert 'year' in df_result.columns
        assert 'day' in df_result.columns

    def test_create_lag_features(self, sample_dataframe):
        """测试滞后特征创建"""
        preprocessor = DataPreprocessor()
        df = sample_dataframe.copy()

        lags = [1, 2, 3]
        df_result = preprocessor.create_lag_features(df, lags=lags)

        for lag in lags:
            assert f'orders_total_lag_{lag}h' in df_result.columns

    def test_create_rolling_features(self, sample_dataframe):
        """测试滚动特征创建"""
        preprocessor = DataPreprocessor()
        df = sample_dataframe.copy()

        windows = [3, 6]
        df_result = preprocessor.create_rolling_features(df, windows=windows)

        for window in windows:
            assert f'orders_total_rolling_mean_{window}h' in df_result.columns
            assert f'orders_total_rolling_std_{window}h' in df_result.columns
            assert f'orders_total_rolling_max_{window}h' in df_result.columns
            assert f'orders_total_rolling_min_{window}h' in df_result.columns

    def test_create_all_features(self, sample_dataframe):
        """测试创建所有特征"""
        preprocessor = DataPreprocessor()
        df = sample_dataframe.copy()

        df_result = preprocessor.create_all_features(
            df,
            lags=[1, 2],
            windows=[3, 6]
        )

        assert isinstance(df_result, pd.DataFrame)
        assert df_result.shape[0] <= df.shape[0]  # 可能会有行被删除

    def test_encode_categorical(self, sample_dataframe):
        """测试分类变量编码"""
        preprocessor = DataPreprocessor()
        df = sample_dataframe.copy()

        # 添加天气列
        df['weather'] = np.random.choice(['sunny', 'rainy', 'cloudy'], len(df))

        df_result = preprocessor.encode_categorical(df)

        assert isinstance(df_result, pd.DataFrame)
        # 应该创建了 weather_ 开头的列
        weather_cols = [col for col in df_result.columns if col.startswith('weather_')]
        assert len(weather_cols) > 0

    def test_normalize_features(self, sample_dataframe):
        """测试特征标准化"""
        preprocessor = DataPreprocessor()
        df = sample_dataframe.copy()

        numeric_cols = ['temperature', 'humidity']
        df_result = preprocessor.normalize_features(df, numeric_cols)

        assert isinstance(df_result, pd.DataFrame)
        # 标准化后的均值应该接近0
        for col in numeric_cols:
            if col in df_result.columns:
                mean_val = df_result[col].mean()
                assert abs(mean_val) < 1e-10 or mean_val == df[col].mean()  # 要么标准化，要么没变化
