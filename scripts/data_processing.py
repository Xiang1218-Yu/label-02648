#!/usr/bin/env python
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
spec = importlib.util.spec_from_file_location("config_module", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.py'))
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
Config = config_module.Config

import pandas as pd
from src.data_preprocessing import DataPreprocessor
from data.data_generator import generate_simulated_data

def generate_data(days=None):
    if days is None:
        days = Config.SIMULATION_DAYS
    
    print(f"Generating simulated data for {days} days...")
    df = generate_simulated_data(days=days)
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    df.to_csv(Config.DATA_PATH, index=False)
    print(f"Data saved to {Config.DATA_PATH}")
    print(f"Data shape: {df.shape}")
    return df

def preprocess_data(input_path=None, output_path=None):
    if input_path is None:
        input_path = Config.DATA_PATH
    
    if output_path is None:
        output_path = os.path.join(Config.DATA_DIR, 'preprocessed_data.csv')
    
    print(f"Preprocessing data from {input_path}...")
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_all_features(
        df_clean,
        lags=Config.LAG_FEATURES,
        windows=Config.ROLLING_WINDOWS
    )
    
    df_features.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    print(f"Preprocessed data shape: {df_features.shape}")
    return df_features

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data processing tools')
    parser.add_argument('--generate', type=int, help='Generate N days of simulated data')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess existing data')
    parser.add_argument('--input', type=str, help='Input CSV path for preprocessing')
    parser.add_argument('--output', type=str, help='Output CSV path')
    
    args = parser.parse_args()
    
    if args.generate:
        generate_data(args.generate)
    
    if args.preprocess or (args.input or args.output):
        preprocess_data(args.input, args.output)
    
    if not any(vars(args).values()):
        parser.print_help()
