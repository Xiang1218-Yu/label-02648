"""
工具函数模块
"""
import numpy as np
import pandas as pd


def convert_to_json_serializable(obj):
    """
    递归转换对象为 JSON 可序列化的类型
    
    参数:
        obj: 任意对象
    
    返回:
        JSON 可序列化的对象
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif pd.isna(obj):
        return None
    else:
        return obj


def safe_int(value):
    """
    安全地转换为 int 类型
    
    参数:
        value: 任意数值
    
    返回:
        int 类型的值
    """
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (float, np.floating)):
        return int(value)
    else:
        return int(value)


def safe_float(value):
    """
    安全地转换为 float 类型
    
    参数:
        value: 任意数值
    
    返回:
        float 类型的值
    """
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, (int, np.integer)):
        return float(value)
    else:
        return float(value)
