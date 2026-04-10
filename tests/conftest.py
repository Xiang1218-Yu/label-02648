#!/usr/bin/env python
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def sample_data():
    dates = pd.date_range(start="2024-01-01", periods=24 * 7, freq="h")
    n = len(dates)

    df = pd.DataFrame(
        {
            "datetime": dates,
            "orders_total": np.random.randint(50, 200, n),
            "hour": dates.hour,
            "day_of_week": dates.dayofweek,
            "month": dates.month,
            "is_weekend": (dates.dayofweek >= 5).astype(int),
            "region_a_orders": np.random.randint(10, 50, n),
            "region_b_orders": np.random.randint(10, 50, n),
            "region_c_orders": np.random.randint(10, 50, n),
            "temperature": np.random.uniform(10, 35, n),
            "weather_condition": np.random.choice([0, 1, 2, 3], n),
        }
    )

    return df


@pytest.fixture(scope="function")
def config_fixture():
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "config_module",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py"
        ),
    )
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.Config
