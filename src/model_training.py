"""
模型训练模块
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        
    def train_random_forest(self, X_train, y_train, use_grid_search=True):
        """训练随机森林模型"""
        logger.info("开始训练随机森林模型...")
        
        if use_grid_search:
            # 定义参数网格
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # 网格搜索
            rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='r2', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            self.best_params['random_forest'] = grid_search.best_params_
            logger.info(f"最佳参数: {grid_search.best_params_}")
            logger.info(f"最佳CV R2分数: {grid_search.best_score_:.4f}")
        else:
            # 使用默认参数快速训练
            best_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            best_model.fit(X_train, y_train)
            logger.info("使用默认参数训练完成")
        
        self.models['random_forest'] = best_model
        logger.info("随机森林模型训练完成")
        return best_model
    
    def train_xgboost(self, X_train, y_train, use_grid_search=True):
        """训练XGBoost模型"""
        logger.info("开始训练XGBoost模型...")
        
        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=5, scoring='r2',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            self.best_params['xgboost'] = grid_search.best_params_
            logger.info(f"最佳参数: {grid_search.best_params_}")
            logger.info(f"最佳CV R2分数: {grid_search.best_score_:.4f}")
        else:
            # 使用默认参数快速训练
            best_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                n_jobs=-1
            )
            best_model.fit(X_train, y_train)
            logger.info("使用默认参数训练完成")
        
        self.models['xgboost'] = best_model
        logger.info("XGBoost模型训练完成")
        return best_model
    
    def train_lightgbm(self, X_train, y_train, use_grid_search=False):
        """训练LightGBM模型（备选）"""
        logger.info("开始训练LightGBM模型...")
        
        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 70]
            }
            
            lgb_model = lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1)
            grid_search = GridSearchCV(
                lgb_model, param_grid, cv=5, scoring='r2',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            self.best_params['lightgbm'] = grid_search.best_params_
            logger.info(f"最佳参数: {grid_search.best_params_}")
        else:
            best_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                num_leaves=50,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            best_model.fit(X_train, y_train)
            logger.info("使用默认参数训练完成")
        
        self.models['lightgbm'] = best_model
        logger.info("LightGBM模型训练完成")
        return best_model
    
    def train_all_models(self, X_train, y_train, models_to_train=['random_forest', 'xgboost'], use_grid_search=False):
        """训练所有指定的模型"""
        logger.info(f"开始训练模型: {models_to_train}")
        
        for model_name in models_to_train:
            if model_name == 'random_forest':
                self.train_random_forest(X_train, y_train, use_grid_search)
            elif model_name == 'xgboost':
                self.train_xgboost(X_train, y_train, use_grid_search)
            elif model_name == 'lightgbm':
                self.train_lightgbm(X_train, y_train, use_grid_search)
            else:
                logger.warning(f"未知模型: {model_name}")
        
        logger.info(f"所有模型训练完成，共训练 {len(self.models)} 个模型")
        return self.models
    
    def save_models(self, save_dir='models'):
        """保存训练好的模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(save_dir, f'{name}_model.pkl')
            joblib.dump(model, filepath)
            logger.info(f"模型已保存: {filepath}")
    
    def load_models(self, load_dir='models'):
        """加载已保存的模型"""
        for filename in os.listdir(load_dir):
            if filename.endswith('_model.pkl'):
                model_name = filename.replace('_model.pkl', '')
                filepath = os.path.join(load_dir, filename)
                self.models[model_name] = joblib.load(filepath)
                logger.info(f"模型已加载: {filepath}")
        
        return self.models

if __name__ == "__main__":
    # 测试模型训练模块
    import sys
    sys.path.append('..')
    from data.data_generator import generate_simulated_data
    from src.data_preprocessing import DataPreprocessor
    from src.feature_engineering import create_advanced_features
    from sklearn.model_selection import train_test_split
    
    logging.basicConfig(level=logging.INFO)
    
    # 生成和预处理数据
    df = generate_simulated_data(days=30)
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_all_features(df_clean)
    df_advanced = create_advanced_features(df_features)
    
    # 准备训练数据
    feature_cols = [col for col in df_advanced.columns 
                    if col not in ['datetime', 'orders_total'] and 
                    not col.startswith('region_') and
                    not col.startswith('busiest_')]
    
    X = df_advanced[feature_cols]
    y = df_advanced['orders_total']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # 训练模型
    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train, y_train, use_grid_search=False)
    
    print(f"\n训练完成，模型数量: {len(models)}")
