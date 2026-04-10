#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
fi

echo "Starting model training..."
python -c "
from src.model_training import ModelTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

Config.create_directories()

df = pd.read_csv(Config.DATA_PATH)
X = df.drop(['orders_total', 'datetime'], axis=1, errors='ignore')
y = df['orders_total']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, shuffle=False
)

trainer = ModelTrainer(random_state=Config.RANDOM_STATE)
models = trainer.train_all_models(X_train, y_train)
trainer.save_models(Config.MODELS_DIR)
print('Models trained and saved successfully!')
"
