#!/usr/bin/env python
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.model_training import ModelTrainer


class TestModelTrainer:
    def test_initialization(self):
        trainer = ModelTrainer(random_state=42)
        assert trainer is not None
        assert trainer.random_state == 42
        assert trainer.models == {}
        assert trainer.best_params == {}

    def _get_sample_xy(self, sample_data):
        X = sample_data.drop(["datetime", "orders_total"], axis=1, errors="ignore")
        y = sample_data["orders_total"]
        X = X.select_dtypes(include=[np.number])
        return X[:50], y[:50]  # Use smaller subset for faster tests

    def test_train_random_forest_with_grid_search(self, sample_data):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)

        model = trainer.train_random_forest(X, y, use_grid_search=True)
        assert model is not None
        assert "random_forest" in trainer.models
        assert "random_forest" in trainer.best_params

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_random_forest_without_grid_search(self, sample_data):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)

        model = trainer.train_random_forest(X, y, use_grid_search=False)
        assert model is not None
        assert "random_forest" in trainer.models

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_xgboost(self, sample_data):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)

        model = trainer.train_xgboost(X, y, use_grid_search=False)
        assert model is not None
        assert "xgboost" in trainer.models

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_xgboost_with_grid_search(self, sample_data):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)

        model = trainer.train_xgboost(X, y, use_grid_search=True)
        assert model is not None
        assert "xgboost" in trainer.models
        assert "xgboost" in trainer.best_params

    def test_train_lightgbm(self, sample_data):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)

        model = trainer.train_lightgbm(X, y, use_grid_search=False)
        assert model is not None
        assert "lightgbm" in trainer.models

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_lightgbm_with_grid_search(self, sample_data):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)

        model = trainer.train_lightgbm(X, y, use_grid_search=True)
        assert model is not None
        assert "lightgbm" in trainer.models
        assert "lightgbm" in trainer.best_params

    def test_train_all_models_rf_only(self, sample_data):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)

        models = trainer.train_all_models(
            X, y, models_to_train=["random_forest"], use_grid_search=False
        )
        assert isinstance(models, dict)
        assert "random_forest" in models
        assert len(models) == 1

    def test_train_all_models_multiple(self, sample_data):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)

        models = trainer.train_all_models(
            X, y, models_to_train=["random_forest", "xgboost"], use_grid_search=False
        )
        assert isinstance(models, dict)
        assert "random_forest" in models
        assert "xgboost" in models
        assert len(models) == 2

    def test_train_all_models_empty_list(self, sample_data):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)

        models = trainer.train_all_models(X, y, models_to_train=[])
        assert isinstance(models, dict)
        assert len(models) == 0

    def test_train_all_models_unknown_model(self, sample_data, caplog):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)

        models = trainer.train_all_models(X, y, models_to_train=["unknown_model"])
        assert len(models) == 0
        # Check that a warning was logged
        found_warning = any(
            "未知模型" in rec.message or "未知模型" in str(rec) for rec in caplog.records
        )

    def test_save_models(self, sample_data):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)
        trainer.train_random_forest(X, y, use_grid_search=False)

        temp_dir = tempfile.mkdtemp()
        try:
            trainer.save_models(save_dir=temp_dir)

            saved_files = os.listdir(temp_dir)
            assert "random_forest_model.pkl" in saved_files
            assert (
                os.path.getsize(os.path.join(temp_dir, "random_forest_model.pkl")) > 0
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_save_models_creates_directory(self, sample_data):
        trainer = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)
        trainer.train_random_forest(X, y, use_grid_search=False)

        temp_dir = tempfile.mkdtemp()
        nested_dir = os.path.join(temp_dir, "nested", "models")

        try:
            trainer.save_models(save_dir=nested_dir)
            assert os.path.exists(nested_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_models(self, sample_data):
        # First train and save models
        trainer1 = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)
        trainer1.train_random_forest(X, y, use_grid_search=False)

        temp_dir = tempfile.mkdtemp()
        try:
            trainer1.save_models(save_dir=temp_dir)

            # Now load into a new trainer
            trainer2 = ModelTrainer()
            loaded_models = trainer2.load_models(load_dir=temp_dir)

            assert "random_forest" in loaded_models
            assert len(loaded_models) == 1

            # Verify model can still make predictions
            predictions = loaded_models["random_forest"].predict(X)
            assert len(predictions) == len(y)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_models_directory_not_exists(self):
        trainer = ModelTrainer()
        with pytest.raises(FileNotFoundError):
            trainer.load_models(load_dir="/path/that/does/not/exist")

    def test_multiple_models_saveload_cycle(self, sample_data):
        trainer1 = ModelTrainer(random_state=42)
        X, y = self._get_sample_xy(sample_data)
        trainer1.train_all_models(
            X, y, models_to_train=["random_forest", "xgboost"], use_grid_search=False
        )

        temp_dir = tempfile.mkdtemp()
        try:
            trainer1.save_models(save_dir=temp_dir)

            trainer2 = ModelTrainer()
            loaded_models = trainer2.load_models(load_dir=temp_dir)

            assert "random_forest" in loaded_models
            assert "xgboost" in loaded_models
            assert len(loaded_models) == 2
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
