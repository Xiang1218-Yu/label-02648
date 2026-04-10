import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import create_advanced_features

@pytest.fixture
def feature_test_df():
    dates = pd.date_range('2023-01-01', periods=24*14, freq='h')
    df = pd.DataFrame({
        'hour': dates.hour,
        'weekday': dates.weekday,
        'month': dates.month,
        'is_weekend': (dates.weekday >= 5).astype(int),
        'orders_total': np.random.randint(10, 100, size=len(dates)),
        'temperature': np.random.uniform(10, 35, size=len(dates)),
        'weather': np.random.randint(0, 3, size=len(dates))
    })
    return df

def test_create_advanced_features(feature_test_df):
    df_features = create_advanced_features(feature_test_df)
    assert len(df_features) > 0
    
def test_periodic_features(feature_test_df):
    df_features = create_advanced_features(feature_test_df)
    assert 'hour_sin' in df_features.columns
    assert 'hour_cos' in df_features.columns
    assert 'weekday_sin' in df_features.columns
    assert 'weekday_cos' in df_features.columns
    assert 'month_sin' in df_features.columns
    assert 'month_cos' in df_features.columns

def test_interaction_features(feature_test_df):
    df_features = create_advanced_features(feature_test_df)
    assert 'hour_weekend_interaction' in df_features.columns
    assert 'temp_weather_interaction' in df_features.columns

def test_trend_features(feature_test_df):
    df_features = create_advanced_features(feature_test_df)
    assert 'order_trend' in df_features.columns
    assert 'order_acceleration' in df_features.columns
    assert 'order_change_rate' in df_features.columns
