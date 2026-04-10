import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor

@pytest.fixture
def preprocessor():
    return DataPreprocessor()

@pytest.fixture
def sample_df():
    dates = pd.date_range('2023-01-01', periods=24*7, freq='h')
    df = pd.DataFrame({
        'datetime': dates,
        'orders_total': np.random.randint(10, 100, size=len(dates)),
        'weather': np.random.choice(['sunny', 'rainy', 'cloudy'], size=len(dates)),
        'hour': dates.hour,
        'weekday': dates.weekday,
        'is_weekend': (dates.weekday >= 5).astype(int)
    })
    return df

def test_load_data(preprocessor, sample_data_path):
    df = preprocessor.load_data(sample_data_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'datetime' in df.columns
    assert 'orders_total' in df.columns or 'order_count' in df.columns

def test_clean_data(preprocessor, sample_df):
    sample_df.loc[10, 'orders_total'] = None
    cleaned_df = preprocessor.clean_data(sample_df)
    null_counts = cleaned_df.select_dtypes(include=[np.number]).isnull().sum().sum()
    assert null_counts == 0

def test_create_time_features(preprocessor, sample_df):
    df_time = preprocessor.create_time_features(sample_df)
    assert 'year' in df_time.columns
    assert 'month' in df_time.columns
    assert 'day' in df_time.columns

def test_create_lag_features(preprocessor, sample_df):
    df_lag = preprocessor.create_lag_features(sample_df)
    expected_cols = [f'orders_total_lag_{h}h' for h in [1, 2, 3, 6, 12]]
    for col in expected_cols:
        assert col in df_lag.columns

def test_create_rolling_features(preprocessor, sample_df):
    df_roll = preprocessor.create_rolling_features(sample_df)
    for window in [3, 6, 12, 24]:
        assert f'orders_total_rolling_mean_{window}h' in df_roll.columns
        assert f'orders_total_rolling_std_{window}h' in df_roll.columns

def test_encode_categorical(preprocessor, sample_df):
    df_encoded = preprocessor.encode_categorical(sample_df)
    has_weather_encoded = any('weather_' in col for col in df_encoded.columns)
    expected_onehot = len(sample_df['weather'].unique())
    actual_onehot = sum(1 for col in df_encoded.columns if col.startswith('weather_'))
    if actual_onehot > 0:
        assert actual_onehot == expected_onehot
    assert 'weather' in df_encoded.columns

def test_normalize_features(preprocessor, sample_df):
    features_to_scale = ['orders_total']
    df_norm = preprocessor.normalize_features(sample_df, features_to_scale)
    assert abs(df_norm['orders_total'].mean()) < 2.0

def test_create_all_features(preprocessor, sample_df):
    df_features = preprocessor.create_all_features(sample_df)
    assert len(df_features) > 0
