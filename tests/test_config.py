"""
配置模块测试
"""
import os
from pathlib import Path
import pytest
import tempfile
from config import Config, _parse_bool, _parse_int, _parse_float, _parse_int_list, _parse_float_tuple


class TestParsers:
    """测试解析函数"""
    
    def test_parse_bool(self):
        """测试布尔值解析"""
        assert _parse_bool('true', False) is True
        assert _parse_bool('True', False) is True
        assert _parse_bool('1', False) is True
        assert _parse_bool('yes', False) is True
        assert _parse_bool('on', False) is True
        assert _parse_bool('false', True) is False
        assert _parse_bool(None, True) is True
        assert _parse_bool(None, False) is False
    
    def test_parse_int(self):
        """测试整数解析"""
        assert _parse_int('42', 0) == 42
        assert _parse_int('-10', 0) == -10
        assert _parse_int(None, 100) == 100
        assert _parse_int('invalid', 100) == 100
    
    def test_parse_float(self):
        """测试浮点数解析"""
        assert _parse_float('3.14', 0.0) == 3.14
        assert _parse_float('-0.5', 0.0) == -0.5
        assert _parse_float(None, 1.5) == 1.5
        assert _parse_float('invalid', 1.5) == 1.5
    
    def test_parse_int_list(self):
        """测试整数列表解析"""
        assert _parse_int_list('1,2,3', []) == [1, 2, 3]
        assert _parse_int_list('1, 2, 3', []) == [1, 2, 3]
        assert _parse_int_list(None, [1, 2]) == [1, 2]
        assert _parse_int_list('invalid', [1, 2]) == [1, 2]
    
    def test_parse_float_tuple(self):
        """测试浮点数元组解析"""
        assert _parse_float_tuple('12,8', (10, 6)) == (12.0, 8.0)
        assert _parse_float_tuple('12, 8', (10, 6)) == (12.0, 8.0)
        assert _parse_float_tuple(None, (10, 6)) == (10, 6)
        assert _parse_float_tuple('invalid', (10, 6)) == (10, 6)


class TestConfig:
    """测试 Config 类"""
    
    def test_base_dir_exists(self):
        """测试 BASE_DIR 是否存在"""
        assert Config.BASE_DIR.exists()
        assert Config.BASE_DIR.is_dir()
    
    def test_directory_paths_are_path_objects(self):
        """测试目录路径是 Path 对象"""
        assert isinstance(Config.DATA_DIR, Path)
        assert isinstance(Config.MODELS_DIR, Path)
        assert isinstance(Config.RESULTS_DIR, Path)
        assert isinstance(Config.LOGS_DIR, Path)
        assert isinstance(Config.DATA_PATH, Path)
    
    def test_data_path(self):
        """测试数据路径配置"""
        assert Config.DATA_PATH == Config.DATA_DIR / 'simulated_data.csv'
    
    def test_model_config(self):
        """测试模型配置"""
        assert isinstance(Config.MODELS_TO_TRAIN, list)
        assert len(Config.MODELS_TO_TRAIN) > 0
        assert isinstance(Config.TEST_SIZE, float)
        assert 0 < Config.TEST_SIZE < 1
        assert isinstance(Config.RANDOM_STATE, int)
        assert isinstance(Config.CV_FOLDS, int)
        assert Config.CV_FOLDS > 0
    
    def test_feature_config(self):
        """测试特征工程配置"""
        assert isinstance(Config.LAG_FEATURES, list)
        assert isinstance(Config.ROLLING_WINDOWS, list)
        assert all(isinstance(x, int) for x in Config.LAG_FEATURES)
        assert all(isinstance(x, int) for x in Config.ROLLING_WINDOWS)
    
    def test_create_directories(self):
        """测试创建目录方法"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # 临时修改配置
            original_data_dir = Config.DATA_DIR
            original_models_dir = Config.MODELS_DIR
            original_results_dir = Config.RESULTS_DIR
            original_logs_dir = Config.LOGS_DIR
            
            Config.DATA_DIR = tmp_path / 'data'
            Config.MODELS_DIR = tmp_path / 'models'
            Config.RESULTS_DIR = tmp_path / 'results'
            Config.LOGS_DIR = tmp_path / 'logs'
            
            try:
                Config.create_directories()
                
                assert Config.DATA_DIR.exists()
                assert Config.MODELS_DIR.exists()
                assert Config.RESULTS_DIR.exists()
                assert Config.LOGS_DIR.exists()
            finally:
                # 恢复原始配置
                Config.DATA_DIR = original_data_dir
                Config.MODELS_DIR = original_models_dir
                Config.RESULTS_DIR = original_results_dir
                Config.LOGS_DIR = original_logs_dir
    
    def test_log_config(self):
        """测试日志配置"""
        assert hasattr(Config, 'LOG_LEVEL')
        assert hasattr(Config, 'LOG_FORMAT')
        assert Config.LOG_LEVEL in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    def test_environment_config(self):
        """测试环境配置"""
        assert hasattr(Config, 'ENVIRONMENT')
        assert hasattr(Config, 'DEBUG')
        assert isinstance(Config.DEBUG, bool)
    
    def test_to_dict(self):
        """测试配置转字典"""
        config_dict = Config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'BASE_DIR' in config_dict
        assert 'DATA_DIR' in config_dict
        assert 'MODELS_TO_TRAIN' in config_dict
