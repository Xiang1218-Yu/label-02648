"""测试配置模块"""

import pytest


class TestConfig:
    def test_default_environment(self, test_config):
        """测试默认环境应为测试环境"""
        assert test_config.ENVIRONMENT.value == "test"

    def test_test_environment_settings(self, test_config):
        """测试环境应有特定的设置"""
        assert test_config.SAVE_RESULTS is False
        assert test_config.SIMULATION_DAYS == 7
        assert test_config.USE_OPTUNA is False

    def test_path_settings(self, test_config):
        """测试路径配置"""
        assert test_config.BASE_DIR is not None
        assert test_config.DATA_DIR is not None
        assert test_config.MODELS_DIR is not None
        assert test_config.RESULTS_DIR is not None

    @pytest.mark.parametrize("env_name", ["development", "production", "test"])
    def test_all_environments_loadable(self, env_name):
        """测试所有环境都可正确加载"""
        from config import get_config

        config = get_config(env_name)
        assert config is not None
        assert config.ENVIRONMENT is not None

    def test_env_variable_override(self, monkeypatch):
        """测试环境变量覆盖"""
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")

        from config import get_config

        config = get_config()
        assert config.ENVIRONMENT.value == "production"

    def test_directories_created(self, test_config):
        """测试目录创建功能"""
        assert test_config.DATA_DIR.exists()
        assert test_config.MODELS_DIR.exists()
        assert test_config.RESULTS_DIR.exists()
        assert test_config.LOGS_DIR.exists()
