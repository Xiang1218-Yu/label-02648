#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    create_advanced_features,
    create_region_features,
    extract_fourier_features,
)


class TestFeatureEngineering:
    @pytest.fixture
    def basic_data(self):
        dates = pd.date_range(start="2024-01-01", periods=168, freq="h")
        return pd.DataFrame(
            {
                "datetime": dates,
                "orders_total": np.random.poisson(100, 168),
                "hour": dates.hour,
                "weekday": dates.dayofweek,
                "month": dates.month,
                "is_weekend": (dates.dayofweek >= 5).astype(int),
                "temperature": np.random.uniform(10, 35, 168),
                "weather": np.random.randint(0, 4, 168),
            }
        )

    @pytest.fixture
    def region_data(self, basic_data):
        basic_data["region_a_orders"] = np.random.randint(10, 50, 168)
        basic_data["region_b_orders"] = np.random.randint(10, 50, 168)
        basic_data["region_c_orders"] = np.random.randint(10, 50, 168)
        return basic_data

    def test_create_advanced_features_creates_cyclic_features(self, basic_data):
        df_advanced = create_advanced_features(basic_data)

        assert "hour_sin" in df_advanced.columns
        assert "hour_cos" in df_advanced.columns
        assert "weekday_sin" in df_advanced.columns
        assert "weekday_cos" in df_advanced.columns
        assert "month_sin" in df_advanced.columns
        assert "month_cos" in df_advanced.columns

        assert np.all(df_advanced["hour_sin"] >= -1) and np.all(
            df_advanced["hour_sin"] <= 1
        )
        assert np.all(df_advanced["hour_cos"] >= -1) and np.all(
            df_advanced["hour_cos"] <= 1
        )

    def test_create_advanced_features_creates_interaction_features(self, basic_data):
        df_advanced = create_advanced_features(basic_data)

        assert "temp_weather_interaction" in df_advanced.columns
        assert "hour_weekend_interaction" in df_advanced.columns
        assert "temp_weekend_interaction" in df_advanced.columns

    def test_create_advanced_features_creates_trend_features(self, basic_data):
        df_advanced = create_advanced_features(basic_data)

        assert "order_trend" in df_advanced.columns
        assert "order_acceleration" in df_advanced.columns
        assert "order_change_rate" in df_advanced.columns

    def test_create_advanced_features_creates_binned_features(self, basic_data):
        df_advanced = create_advanced_features(basic_data)

        hour_bin_cols = [c for c in df_advanced.columns if c.startswith("hour_bin_")]
        assert len(hour_bin_cols) > 0
        temp_bin_cols = [c for c in df_advanced.columns if c.startswith("temp_bin_")]
        assert len(temp_bin_cols) > 0

    def test_extract_fourier_features_basic(self):
        series = pd.Series(np.arange(48))
        features = extract_fourier_features(series, period=24, n_harmonics=2)

        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == 48
        assert "fourier_sin_24_1" in features.columns
        assert "fourier_cos_24_1" in features.columns
        assert "fourier_sin_24_2" in features.columns
        assert "fourier_cos_24_2" in features.columns

    def test_extract_fourier_features_values(self):
        series = pd.Series(np.arange(24))
        features = extract_fourier_features(series, period=24, n_harmonics=1)

        assert np.isclose(features["fourier_sin_24_1"].iloc[0], 0, atol=0.01)
        assert np.isclose(features["fourier_cos_24_1"].iloc[0], 1, atol=0.01)

    def test_create_region_features_with_regions(self, region_data):
        df_region = create_region_features(region_data)

        assert "region_a_ratio" in df_region.columns
        assert "region_b_ratio" in df_region.columns
        assert "region_c_ratio" in df_region.columns

        assert "region_variance" in df_region.columns
        assert "busiest_region" in df_region.columns
        assert "region_concentration" in df_region.columns

    def test_create_region_features_ratios(self, region_data):
        df_region = create_region_features(region_data)

        assert np.all(df_region["region_a_ratio"] >= 0)
        assert np.all(df_region["region_a_ratio"] <= 1)

    def test_create_region_features_no_regions(self, basic_data):
        df_no_regions = create_region_features(basic_data)
        assert df_no_regions is not None

    def test_feature_count_increases(self, basic_data):
        initial_count = basic_data.shape[1]
        df_advanced = create_advanced_features(basic_data)
        final_count = df_advanced.shape[1]
        assert final_count > initial_count
