import pytest
import numpy as np
import pandas as pd
from src.utils import convert_to_json_serializable, safe_int, safe_float

def test_convert_numpy_int():
    result = convert_to_json_serializable(np.int64(42))
    assert result == 42
    assert isinstance(result, int)

def test_convert_numpy_float():
    result = convert_to_json_serializable(np.float64(3.14))
    assert result == 3.14
    assert isinstance(result, float)

def test_convert_numpy_array():
    arr = np.array([1, 2, 3])
    result = convert_to_json_serializable(arr)
    assert result == [1, 2, 3]
    assert isinstance(result, list)

def test_convert_dict():
    test_dict = {'a': np.int64(1), 'b': np.float64(2.5)}
    result = convert_to_json_serializable(test_dict)
    assert result == {'a': 1, 'b': 2.5}
    assert isinstance(result['a'], int)
    assert isinstance(result['b'], float)

def test_convert_list():
    test_list = [np.int64(1), np.float64(2.5)]
    result = convert_to_json_serializable(test_list)
    assert result == [1, 2.5]
    assert isinstance(result[0], int)
    assert isinstance(result[1], float)

def test_convert_pandas_series():
    series = pd.Series([1, 2, 3])
    result = convert_to_json_serializable(series)
    assert isinstance(result, list)
    assert len(result) == 3

def test_safe_int_from_numpy():
    result = safe_int(np.int64(100))
    assert result == 100
    assert isinstance(result, int)

def test_safe_int_from_float():
    result = safe_int(123.456)
    assert result == 123
    assert isinstance(result, int)

def test_safe_float_from_numpy():
    result = safe_float(np.float64(3.14))
    assert result == 3.14
    assert isinstance(result, float)

def test_safe_float_from_int():
    result = safe_float(42)
    assert result == 42.0
    assert isinstance(result, float)
