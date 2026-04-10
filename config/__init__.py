# Package marker for config module
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import Config as _RootConfig

    _has_old_config = True
except ImportError:
    _has_old_config = False

from .loader import Config as _NewConfig
from .loader import get_config

if _has_old_config:
    Config = _RootConfig
else:
    Config = _NewConfig

__all__ = ["get_config", "Config"]
