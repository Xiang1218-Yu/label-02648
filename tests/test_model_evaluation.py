#!/usr/bin/env python
import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.model_evaluation import ModelEvaluator


class TestModelEvaluator:
    @pytest.fixture
    def sample_model_data(self):
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1
        return X, y

    @pytest.fixture
    def trained_models(self, sample_model_data):
        X, y = sample_model_data
        lr = LinearRegression()
        lr.fit(X, y)
        dt = DecisionTreeRegressor(max_depth=3, random_state=42)
        dt.fit(X, y)
        return {"LinearRegression": lr, "DecisionTree": dt}

    def test_initialization(self, trained_models):
        evaluator = ModelEvaluator(trained_models)
        assert evaluator is not None
        assert evaluator.models is not None
        assert evaluator.metrics == {}
        assert evaluator.predictions == {}

    def test_calculate_mape(self, trained_models):
        evaluator = ModelEvaluator(trained_models)
        y_true = np.array([100, 200, 300, 0, 500])
        y_pred = np.array([110, 190, 330, 10, 480])

        mape = evaluator.calculate_mape(y_true, y_pred)
        assert isinstance(mape, float)
        assert mape > 0

    def test_calculate_metrics(self, trained_models, sample_model_data):
        X, y = sample_model_data
        evaluator = ModelEvaluator(trained_models)
        metrics = evaluator.calculate_metrics(X, y)

        assert isinstance(metrics, dict)
        assert "LinearRegression" in metrics
        assert "DecisionTree" in metrics

        for model_name, model_metrics in metrics.items():
            assert "MAE" in model_metrics
            assert "MSE" in model_metrics
            assert "RMSE" in model_metrics
            assert "R2" in model_metrics
            assert "MAPE" in model_metrics
            assert "Explained_Variance" in model_metrics

    def test_predictions_stored(self, trained_models, sample_model_data):
        X, y = sample_model_data
        evaluator = ModelEvaluator(trained_models)
        evaluator.calculate_metrics(X, y)

        assert "LinearRegression" in evaluator.predictions
        assert "DecisionTree" in evaluator.predictions

    def test_plot_predictions(self, trained_models, sample_model_data):
        X, y = sample_model_data
        evaluator = ModelEvaluator(trained_models)
        evaluator.calculate_metrics(X, y)
        fig = evaluator.plot_predictions(X, y)
        assert fig is not None
        matplotlib.pyplot.close()

    def test_plot_predictions_save_path(
        self, tmp_path, trained_models, sample_model_data
    ):
        X, y = sample_model_data
        save_path = str(tmp_path / "predictions.png")
        evaluator = ModelEvaluator(trained_models)
        evaluator.calculate_metrics(X, y)
        evaluator.plot_predictions(X, y, save_path=save_path)
        assert os.path.exists(save_path)
        matplotlib.pyplot.close()

    def test_plot_feature_importance(self, trained_models, sample_model_data):
        X, y = sample_model_data
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        evaluator = ModelEvaluator(trained_models)
        evaluator.calculate_metrics(X, y)
        fig = evaluator.plot_feature_importance(feature_names)
        assert fig is not None
        matplotlib.pyplot.close()

    def test_plot_feature_importance_save_path(
        self, tmp_path, trained_models, sample_model_data
    ):
        X, y = sample_model_data
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        save_path = str(tmp_path / "feature_importance.png")
        evaluator = ModelEvaluator(trained_models)
        evaluator.calculate_metrics(X, y)
        evaluator.plot_feature_importance(feature_names, save_path=save_path)
        assert os.path.exists(save_path)
        matplotlib.pyplot.close()

    def test_plot_time_series_comparison(self, trained_models, sample_model_data):
        X, y = sample_model_data
        evaluator = ModelEvaluator(trained_models)
        evaluator.calculate_metrics(X, y)
        fig = evaluator.plot_time_series_comparison(y)
        assert fig is not None
        matplotlib.pyplot.close()

    def test_export_metrics_to_excel(self, tmp_path, trained_models, sample_model_data):
        X, y = sample_model_data
        filepath = str(tmp_path / "metrics.xlsx")
        evaluator = ModelEvaluator(trained_models)
        evaluator.calculate_metrics(X, y)
        df = evaluator.export_metrics_to_excel(filepath)
        assert isinstance(df, pd.DataFrame)
        assert os.path.exists(filepath)

    def test_single_model_plot_predictions(self, sample_model_data):
        X, y = sample_model_data
        lr = LinearRegression()
        lr.fit(X, y)
        single_model = {"LinearRegression": lr}
        evaluator = ModelEvaluator(single_model)
        evaluator.calculate_metrics(X, y)
        fig = evaluator.plot_predictions(X, y)
        assert fig is not None
        matplotlib.pyplot.close()

    def test_single_model_plot_feature_importance(self, sample_model_data):
        X, y = sample_model_data
        lr = LinearRegression()
        lr.fit(X, y)
        single_model = {"LinearRegression": lr}
        evaluator = ModelEvaluator(single_model)
        evaluator.calculate_metrics(X, y)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        fig = evaluator.plot_feature_importance(feature_names)
        assert fig is not None
        matplotlib.pyplot.close()

    def test_plot_time_series_comparison_save_path(
        self, tmp_path, trained_models, sample_model_data
    ):
        X, y = sample_model_data
        save_path = str(tmp_path / "timeseries.png")
        evaluator = ModelEvaluator(trained_models)
        evaluator.calculate_metrics(X, y)
        fig = evaluator.plot_time_series_comparison(y, save_path=save_path)
        assert fig is not None
        assert os.path.exists(save_path)
        matplotlib.pyplot.close()

    def test_model_without_feature_importances(self, sample_model_data):
        X, y = sample_model_data
        lr = LinearRegression()
        lr.fit(X, y)
        single_model = {"LinearRegression": lr}
        evaluator = ModelEvaluator(single_model)
        evaluator.calculate_metrics(X, y)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        fig = evaluator.plot_feature_importance(feature_names)
        assert fig is not None
        matplotlib.pyplot.close()
