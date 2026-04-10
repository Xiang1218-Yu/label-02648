#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytest
import os
import tempfile
import matplotlib
matplotlib.use('Agg')

from src.eda import EDAAnalyzer


class TestEDAAnalyzer:
    @pytest.fixture
    def sample_csv(self, sample_data):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
            sample_data.to_csv(filepath, index=False)
        yield filepath
        os.unlink(filepath)

    def test_initialization(self):
        eda = EDAAnalyzer()
        assert eda is not None
        assert eda.df is None

    def test_initialization_with_style(self):
        eda = EDAAnalyzer(figsize=(10, 6), style='default')
        assert eda is not None

    def test_load_data(self, sample_csv, sample_data):
        eda = EDAAnalyzer()
        df = eda.load_data(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert eda.df is not None
        assert len(eda.df) == len(sample_data)

    def test_basic_info(self, sample_data):
        eda = EDAAnalyzer()
        eda.df = sample_data
        stats = eda.basic_info()
        assert isinstance(stats, pd.DataFrame)

    def test_plot_time_series(self, sample_data):
        eda = EDAAnalyzer()
        eda.df = sample_data
        fig = eda.plot_time_series(target_col='orders_total')
        assert fig is not None
        matplotlib.pyplot.close()

    def test_plot_distribution(self, sample_data):
        eda = EDAAnalyzer()
        eda.df = sample_data
        fig = eda.plot_distribution(target_col='orders_total')
        assert fig is not None
        matplotlib.pyplot.close()

    def test_plot_correlation_matrix(self, sample_data):
        eda = EDAAnalyzer()
        eda.df = sample_data
        fig = eda.plot_correlation_matrix()
        assert fig is not None
        matplotlib.pyplot.close()

    def test_plot_categorical_analysis(self, sample_data):
        eda = EDAAnalyzer()
        eda.df = sample_data
        fig = eda.plot_categorical_analysis()
        assert fig is not None
        matplotlib.pyplot.close()

    def test_plot_region_analysis_without_regions(self):
        eda = EDAAnalyzer()
        # Create a DataFrame without region columns
        dates = pd.date_range(start="2024-01-01", periods=24, freq="h")
        eda.df = pd.DataFrame({
            "datetime": dates, 
            "orders": np.random.randint(10, 50, len(dates))
        })
        result = eda.plot_region_analysis()
        assert result is None
        matplotlib.pyplot.close()

    def test_plot_region_analysis_with_regions(self, sample_data):
        sample_data['region_a_orders'] = np.random.randint(10, 50, len(sample_data))
        sample_data['region_b_orders'] = np.random.randint(10, 50, len(sample_data))
        eda = EDAAnalyzer()
        eda.df = sample_data
        fig = eda.plot_region_analysis()
        assert fig is not None
        matplotlib.pyplot.close()

    def test_generate_full_report(self, tmp_path, sample_data):
        output_dir = tmp_path / "eda_report"
        eda = EDAAnalyzer()
        eda.df = sample_data
        eda.generate_full_report(output_dir=str(output_dir))
        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, 'time_series.png'))
        assert os.path.exists(os.path.join(output_dir, 'distribution.png'))
        assert os.path.exists(os.path.join(output_dir, 'correlation_matrix.png'))
        assert os.path.exists(os.path.join(output_dir, 'categorical_analysis.png'))
        matplotlib.pyplot.close('all')

    def test_plots_save_path(self, tmp_path, sample_data):
        save_path = str(tmp_path / "test_plot.png")
        eda = EDAAnalyzer()
        eda.df = sample_data
        eda.plot_time_series(save_path=save_path)
        assert os.path.exists(save_path)
        matplotlib.pyplot.close()
