#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytest

from src.utils import convert_to_json_serializable, safe_int, safe_float


class TestUtils:
    def test_convert_to_json_serializable_with_dict(self):
        data = {
            'int': np.int64(10),
            'float': np.float64(10.5),
            'bool': np.bool_(True),
            'array': np.array([1, 2, 3]),
        }
        
        result = convert_to_json_serializable(data)
        assert isinstance(result['int'], int)
        assert isinstance(result['float'], float)
        assert isinstance(result['bool'], bool)
        assert isinstance(result['array'], list)

    def test_convert_to_json_serializable_with_list(self):
        data = [np.int64(1), np.float32(2.5)]
        result = convert_to_json_serializable(data)
        
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_convert_to_json_serializable_with_tuple(self):
        data = (np.int64(1), np.int64(2))
        result = convert_to_json_serializable(data)
        
        assert isinstance(result, tuple)
        assert isinstance(result[0], int)

    def test_convert_to_json_serializable_with_series(self):
        series = pd.Series([1, 2, 3])
        result = convert_to_json_serializable(series)
        
        assert isinstance(result, list)

    def test_convert_to_json_serializable_with_dataframe(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = convert_to_json_serializable(df)
        
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_convert_to_json_serializable_with_nan(self):
        result = convert_to_json_serializable(np.nan)
        assert result is None

    def test_convert_to_json_serializable_passthrough(self):
        result = convert_to_json_serializable('hello')
        assert result == 'hello'
        
        result = convert_to_json_serializable(123)
        assert result == 123
        
        result = convert_to_json_serializable(123.456)
        assert result == 123.456

    def test_safe_int_with_numpy_int(self):
        assert safe_int(np.int64(42)) == 42
        assert isinstance(safe_int(np.int64(42)), int)

    def test_safe_int_with_float(self):
        assert safe_int(42.7) == 42
        assert safe_int(np.float64(42.7)) == 42

    def test_safe_int_with_string_number(self):
        assert safe_int('42') == 42

    def test_safe_float_with_numpy_float(self):
        assert safe_float(np.float64(3.14)) == 3.14
        assert isinstance(safe_float(np.float64(3.14)), float)

    def test_safe_float_with_int(self):
        assert safe_float(42) == 42.0
        assert safe_float(np.int64(42)) == 42.0

    def test_safe_float_with_string_number(self):
        assert safe_float('3.14') == 3.14
