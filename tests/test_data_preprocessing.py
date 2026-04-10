#!/usr/bin/env python
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    def test_initialization(self):
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, "scaler")
        assert hasattr(preprocessor, "feature_names")

    def test_load_data(self, sample_data):
        preprocessor = DataPreprocessor()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = f.name
            sample_data.to_csv(filepath, index=False)

        df_loaded = preprocessor.load_data(filepath)
        assert isinstance(df_loaded, pd.DataFrame)
        assert "datetime" in df_loaded.columns
        assert pd.api.types.is_datetime64_any_dtype(df_loaded["datetime"])
        os.unlink(filepath)

    def test_clean_data_with_sample_data(self, sample_data):
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_data(sample_data)

        assert isinstance(cleaned, pd.DataFrame)
        assert "datetime" in cleaned.columns
        assert len(cleaned) > 0

    def test_clean_data_handles_nans(self):
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
        df = pd.DataFrame(
            {
                "datetime": dates,
                "orders_total": [50, 60, np.nan, 80] * 25,
            }
        )

        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_data(df)

        assert cleaned["orders_total"].isna().sum() == 0

    def test_clean_data_handles_outliers(self):
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
        orders = [100] * 100
        orders[0] = 1000
        orders[1] = -100

        df = pd.DataFrame(
            {
                "datetime": dates,
                "orders_total": orders,
            }
        )

        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_data(df)

        assert cleaned["orders_total"].iloc[0] < 1000
        assert cleaned["orders_total"].iloc[1] > -100

    def test_create_time_features(self, sample_data):
        preprocessor = DataPreprocessor()
        df_time = preprocessor.create_time_features(sample_data)

        assert isinstance(df_time, pd.DataFrame)
        assert "year" in df_time.columns
        assert "month" in df_time.columns
        assert "day" in df_time.columns
        assert "hour" in df_time.columns
        assert "weekday" in df_time.columns
        assert "is_weekend" in df_time.columns

    def test_create_lag_features(self, sample_data):
        preprocessor = DataPreprocessor()
        lags = [1, 2, 3]
        df_lag = preprocessor.create_lag_features(
            sample_data, target_col="orders_total", lags=lags
        )

        assert isinstance(df_lag, pd.DataFrame)
        for lag in lags:
            assert f"orders_total_lag_{lag}h" in df_lag.columns

    def test_create_rolling_features(self, sample_data):
        preprocessor = DataPreprocessor()
        windows = [3, 6]
        df_roll = preprocessor.create_rolling_features(
            sample_data, target_col="orders_total", windows=windows
        )

        assert isinstance(df_roll, pd.DataFrame)
        for window in windows:
            assert f"orders_total_rolling_mean_{window}h" in df_roll.columns
            assert f"orders_total_rolling_std_{window}h" in df_roll.columns
            assert f"orders_total_rolling_max_{window}h" in df_roll.columns
            assert f"orders_total_rolling_min_{window}h" in df_roll.columns

    def test_encode_categorical(self):
        df = pd.DataFrame(
            {"weather": [0, 1, 2, 0, 1], "orders_total": [100, 110, 90, 105, 95]}
        )
        preprocessor = DataPreprocessor()
        df_encoded = preprocessor.encode_categorical(df)

        assert isinstance(df_encoded, pd.DataFrame)

    def test_normalize_features(self, sample_data):
        preprocessor = DataPreprocessor()
        features = ["orders_total", "temperature"]
        df_norm = preprocessor.normalize_features(sample_data, features)

        assert isinstance(df_norm, pd.DataFrame)
        for f in features:
            if f in df_norm.columns:
                assert np.isclose(df_norm[f].mean(), 0, atol=0.1)

    def test_create_all_features(self, sample_data):
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_data(sample_data)
        features_df = preprocessor.create_all_features(cleaned)

        assert isinstance(features_df, pd.DataFrame)
        assert features_df.shape[1] >= cleaned.shape[1]
        assert features_df.isna().sum().sum() == 0
