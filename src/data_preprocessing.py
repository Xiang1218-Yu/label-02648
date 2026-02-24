"""
数据预处理模块
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, filepath):
        """加载数据并检查缺失值"""
        logger.info(f"加载数据: {filepath}")
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 检查缺失值
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"发现缺失值:\n{missing[missing > 0]}")
        
        logger.info(f"数据加载完成，形状: {df.shape}")
        return df
    
    def clean_data(self, df):
        """数据清洗：处理缺失值、异常值"""
        logger.info("开始数据清洗...")
        df_clean = df.copy()
        
        # 处理缺失值（前向填充）
        df_clean = df_clean.ffill()
        
        # 处理异常值（使用IQR方法）
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['hour', 'weekday', 'is_weekend', 'is_holiday', 'weather']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # 记录异常值数量
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                if outliers > 0:
                    logger.info(f"{col}: 发现 {outliers} 个异常值")
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        logger.info("数据清洗完成")
        return df_clean
    
    def create_time_features(self, df):
        """从datetime创建时间特征"""
        logger.info("创建时间特征...")
        df_time = df.copy()
        
        if 'datetime' in df_time.columns:
            df_time['year'] = df_time['datetime'].dt.year
            df_time['month'] = df_time['datetime'].dt.month
            df_time['day'] = df_time['datetime'].dt.day
            
            # 如果这些特征不存在，则创建
            if 'hour' not in df_time.columns:
                df_time['hour'] = df_time['datetime'].dt.hour
            if 'weekday' not in df_time.columns:
                df_time['weekday'] = df_time['datetime'].dt.weekday
            if 'is_weekend' not in df_time.columns:
                df_time['is_weekend'] = (df_time['weekday'] >= 5).astype(int)
        
        logger.info(f"时间特征创建完成，新增特征数: {len(['year', 'month', 'day'])}")
        return df_time
    
    def create_lag_features(self, df, target_col='orders_total', lags=[1, 2, 3, 6, 12]):
        """创建滞后特征"""
        logger.info(f"创建滞后特征，滞后期: {lags}")
        df_lag = df.copy()
        
        for lag in lags:
            df_lag[f'{target_col}_lag_{lag}h'] = df_lag[target_col].shift(lag)
        
        logger.info(f"滞后特征创建完成，新增特征数: {len(lags)}")
        return df_lag
    
    def create_rolling_features(self, df, target_col='orders_total', windows=[3, 6, 12, 24]):
        """创建滑动窗口特征"""
        logger.info(f"创建滑动窗口特征，窗口大小: {windows}")
        df_roll = df.copy()
        
        for window in windows:
            df_roll[f'{target_col}_rolling_mean_{window}h'] = df_roll[target_col].rolling(window=window).mean()
            df_roll[f'{target_col}_rolling_std_{window}h'] = df_roll[target_col].rolling(window=window).std()
            df_roll[f'{target_col}_rolling_max_{window}h'] = df_roll[target_col].rolling(window=window).max()
            df_roll[f'{target_col}_rolling_min_{window}h'] = df_roll[target_col].rolling(window=window).min()
        
        logger.info(f"滑动窗口特征创建完成，新增特征数: {len(windows) * 4}")
        return df_roll
    
    def encode_categorical(self, df):
        """对分类变量进行编码"""
        logger.info("编码分类变量...")
        df_encoded = df.copy()
        
        # One-hot编码天气
        if 'weather' in df_encoded.columns:
            weather_dummies = pd.get_dummies(df_encoded['weather'], prefix='weather')
            df_encoded = pd.concat([df_encoded, weather_dummies], axis=1)
        
        logger.info("分类变量编码完成")
        return df_encoded
    
    def normalize_features(self, df, features):
        """特征标准化"""
        logger.info(f"标准化特征，特征数: {len(features)}")
        df_norm = df.copy()
        
        # 只标准化存在的特征
        features_to_scale = [f for f in features if f in df_norm.columns]
        
        if features_to_scale:
            df_norm[features_to_scale] = self.scaler.fit_transform(df_norm[features_to_scale])
            logger.info(f"实际标准化特征数: {len(features_to_scale)}")
        
        return df_norm
    
    def create_all_features(self, df, lags=[1, 2, 3, 6, 12], windows=[3, 6, 12, 24]):
        """创建所有特征的便捷方法"""
        logger.info("开始创建所有特征...")
        
        df_processed = self.create_time_features(df)
        df_processed = self.create_lag_features(df_processed, lags=lags)
        df_processed = self.create_rolling_features(df_processed, windows=windows)
        df_processed = self.encode_categorical(df_processed)
        
        # 删除因滞后和滚动窗口产生的NaN行
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna()
        dropped_rows = initial_rows - len(df_processed)
        
        logger.info(f"特征创建完成，删除 {dropped_rows} 行NaN数据")
        logger.info(f"最终数据形状: {df_processed.shape}")
        
        return df_processed

if __name__ == "__main__":
    # 测试预处理模块
    import sys
    sys.path.append('..')
    from data.data_generator import generate_simulated_data
    
    logging.basicConfig(level=logging.INFO)
    
    df = generate_simulated_data(days=7)
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_all_features(df_clean)
    
    print(f"\n处理后数据形状: {df_features.shape}")
    print(f"\n特征列表:\n{df_features.columns.tolist()}")
