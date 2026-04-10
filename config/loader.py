#!/usr/bin/env python
import copy
import os
from typing import Any, Dict, Optional

import yaml


class Config:
    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, env: str = None, config_dir: str = None):
        if self._config:
            return

        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))

        if env is None:
            env = os.getenv("PROJECT_ENV", "dev").lower()

        self._env = env
        self._config_dir = config_dir

        default_config = self._load_yaml("default.yaml")
        env_config = self._load_yaml(f"{env}.yaml")

        self._config = self._deep_merge(default_config, env_config)
        self._resolve_placeholders()
        self._setup_class_attributes()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(self._config_dir, filename)
        if not os.path.exists(filepath):
            return {}

        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = copy.deepcopy(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def _resolve_placeholders(self):
        project_root = os.path.dirname(self._config_dir)

        def resolve_value(value):
            if isinstance(value, str) and "${PROJECT_ROOT}" in value:
                return value.replace("${PROJECT_ROOT}", project_root)
            return value

        def recursive_resolve(d: Dict[str, Any]):
            for k, v in d.items():
                if isinstance(v, dict):
                    recursive_resolve(v)
                elif isinstance(v, str):
                    d[k] = resolve_value(v)

        recursive_resolve(self._config)

    def _setup_class_attributes(self):
        flat_config = self._flatten_config(self._config)
        for k, v in flat_config.items():
            setattr(self.__class__, k.upper(), v)

    def _flatten_config(
        self, config: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        result = {}
        for k, v in config.items():
            key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                result.update(self._flatten_config(v, key))
            else:
                result[key] = v
        return result

    @property
    def env(self) -> str:
        return self._env

    @property
    def all(self) -> Dict[str, Any]:
        return copy.deepcopy(self._config)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getattr__(self, key: str) -> Any:
        return self.get(key)

    @staticmethod
    def create_directories():
        config = get_config()
        paths = config.get("paths", {})
        for path_key in ["data_dir", "models_dir", "results_dir", "logs_dir"]:
            if path_key in paths:
                os.makedirs(paths[path_key], exist_ok=True)


def get_config(env: str = None, config_dir: str = None) -> Config:
    return Config(env, config_dir)
