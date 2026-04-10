import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_data_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'simulated_data.csv')
