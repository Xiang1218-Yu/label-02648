import pytest
import pandas as pd
from src.data_preprocessing import DataPreprocessor

@pytest.fixture
def preprocessor():
    return DataPreprocessor()

def test_load_data(preprocessor, sample_data_path):
    df = preprocessor.load_data(sample_data_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

def test_clean_data(preprocessor, sample_data_path):
    df = preprocessor.load_data(sample_data_path)
    cleaned_df = preprocessor.clean_data(df)
    assert cleaned_df.isnull().sum().sum() == 0
