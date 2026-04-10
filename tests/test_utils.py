"""
工具函数模块测试
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    convert_to_json_serializable,
    safe_int,
    safe_float,
)


class TestConvertToJsonSerializable:
    """测试 convert_to_json_serializable 函数"""
    
    def test_convert_numpy_int(self):
        """测试转换 numpy 整数"""
        result = convert_to_json_serializable(np.int64(42))
        assert isinstance(result, int)
        assert result == 42
    
    def test_convert_numpy_float(self):
        """测试转换 numpy 浮点数"""
        result = convert_to_json_serializable(np.float64(3.14))
        assert isinstance(result, float)
        assert result == 3.14
    
    def test_convert_numpy_bool(self):
        """测试转换 numpy 布尔值"""
        result = convert_to_json_serializable(np.bool_(True))
        assert isinstance(result, bool)
        assert result is True
    
    def test_convert_numpy_array(self):
        """测试转换 numpy 数组"""
        arr = np.array([1, 2, 3])
        result = convert_to_json_serializable(arr)
        assert isinstance(result, list)
        assert result == [1, 2, 3]
    
    def test_convert_pandas_series(self):
        """测试转换 pandas Series"""
        series = pd.Series([1, 2, 3])
        result = convert_to_json_serializable(series)
        assert isinstance(result, list)
        assert result == [1, 2, 3]
    
    def test_convert_pandas_dataframe(self):
        """测试转换 pandas DataFrame"""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = convert_to_json_serializable(df)
        assert isinstance(result, list)
        assert result == [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
    
    def test_convert_dict(self):
        """测试转换字典"""
        data = {'key': np.int64(42), 'value': np.float64(3.14)}
        result = convert_to_json_serializable(data)
        assert isinstance(result, dict)
        assert result == {'key': 42, 'value': 3.14}
    
    def test_convert_list(self):
        """测试转换列表"""
        data = [np.int64(1), np.int64(2), np.int64(3)]
        result = convert_to_json_serializable(data)
        assert isinstance(result, list)
        assert result == [1, 2, 3]
    
    def test_convert_nested_structure(self):
        """测试转换嵌套结构"""
        data = {
            'array': np.array([1, 2, 3]),
            'nested': {
                'value': np.float64(3.14)
            },
            'list': [np.int64(1), np.int64(2)]
        }
        result = convert_to_json_serializable(data)
        assert result == {
            'array': [1, 2, 3],
            'nested': {'value': 3.14},
            'list': [1, 2]
        }
    
    def test_convert_nan(self):
        """测试转换 NaN"""
        result = convert_to_json_serializable(np.nan)
        assert result is None
    
    def test_convert_regular_types(self):
        """测试转换常规类型"""
        assert convert_to_json_serializable(42) == 42
        assert convert_to_json_serializable(3.14) == 3.14
        assert convert_to_json_serializable("string") == "string"
        assert convert_to_json_serializable(True) is True


class TestSafeInt:
    """测试 safe_int 函数"""
    
    def test_safe_int_from_numpy_int64(self):
        """测试从 numpy int64 转换"""
        result = safe_int(np.int64(42))
        assert isinstance(result, int)
        assert result == 42
    
    def test_safe_int_from_numpy_int32(self):
        """测试从 numpy int32 转换"""
        result = safe_int(np.int32(42))
        assert isinstance(result, int)
        assert result == 42
    
    def test_safe_int_from_float(self):
        """测试从 float 转换"""
        result = safe_int(3.14)
        assert isinstance(result, int)
        assert result == 3
    
    def test_safe_int_from_numpy_float(self):
        """测试从 numpy float 转换"""
        result = safe_int(np.float64(3.14))
        assert isinstance(result, int)
        assert result == 3
    
    def test_safe_int_from_int(self):
        """测试从 int 转换"""
        result = safe_int(42)
        assert isinstance(result, int)
        assert result == 42


class TestSafeFloat:
    """测试 safe_float 函数"""
    
    def test_safe_float_from_numpy_float64(self):
        """测试从 numpy float64 转换"""
        result = safe_float(np.float64(3.14))
        assert isinstance(result, float)
        assert result == 3.14
    
    def test_safe_float_from_numpy_float32(self):
        """测试从 numpy float32 转换"""
        result = safe_float(np.float32(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.01
    
    def test_safe_float_from_int(self):
        """测试从 int 转换"""
        result = safe_float(42)
        assert isinstance(result, float)
        assert result == 42.0
    
    def test_safe_float_from_numpy_int(self):
        """测试从 numpy int 转换"""
        result = safe_float(np.int64(42))
        assert isinstance(result, float)
        assert result == 42.0
    
    def test_safe_float_from_float(self):
        """测试从 float 转换"""
        result = safe_float(3.14)
        assert isinstance(result, float)
        assert result == 3.14
