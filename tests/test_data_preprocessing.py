#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    def test_initialization(self):
        preprocessor = DataPreprocessor()
        assert preprocessor is not None

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

    def test_create_all_features(self, sample_data):
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_data(sample_data)
        features_df = preprocessor.create_all_features(cleaned)

        assert isinstance(features_df, pd.DataFrame)
        assert features_df.shape[1] >= cleaned.shape[1]
