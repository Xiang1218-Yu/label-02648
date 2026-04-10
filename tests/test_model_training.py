#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytest

from src.model_training import ModelTrainer


class TestModelTrainer:
    def test_initialization(self):
        trainer = ModelTrainer(random_state=42)
        assert trainer is not None
        assert trainer.random_state == 42

    def test_train_single_model(self, sample_data):
        trainer = ModelTrainer(random_state=42)

        X = sample_data.drop(["datetime", "orders_total"], axis=1, errors="ignore")
        y = sample_data["orders_total"]

        X = X.select_dtypes(include=[np.number])

        model = trainer.train_random_forest(X, y)
        assert model is not None

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_available_models(self):
        trainer = ModelTrainer()
        models = trainer.train_all_models(None, None, models_to_train=[])
        assert isinstance(models, dict)
