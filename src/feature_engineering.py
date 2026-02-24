"""
特征工程模块 - 创建高级特征
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_advanced_features(df):
    """
    创建高级特征
    
    包括：
    1. 周期性特征（傅里叶变换）
    2. 交互特征
    3. 统计特征
    4. 分箱特征
    """
    logger.info("开始创建高级特征...")
    df_advanced = df.copy()
    
    # 1. 周期性特征（使用三角函数编码）
    if 'hour' in df_advanced.columns:
        df_advanced['hour_sin'] = np.sin(2 * np.pi * df_advanced['hour'] / 24)
        df_advanced['hour_cos'] = np.cos(2 * np.pi * df_advanced['hour'] / 24)
    
    if 'weekday' in df_advanced.columns:
        df_advanced['weekday_sin'] = np.sin(2 * np.pi * df_advanced['weekday'] / 7)
        df_advanced['weekday_cos'] = np.cos(2 * np.pi * df_advanced['weekday'] / 7)
    
    if 'month' in df_advanced.columns:
        df_advanced['month_sin'] = np.sin(2 * np.pi * df_advanced['month'] / 12)
        df_advanced['month_cos'] = np.cos(2 * np.pi * df_advanced['month'] / 12)
    
    # 2. 交互特征
    if 'temperature' in df_advanced.columns and 'weather' in df_advanced.columns:
        df_advanced['temp_weather_interaction'] = df_advanced['temperature'] * df_advanced['weather']
    
    if 'hour' in df_advanced.columns and 'is_weekend' in df_advanced.columns:
        df_advanced['hour_weekend_interaction'] = df_advanced['hour'] * df_advanced['is_weekend']
    
    if 'temperature' in df_advanced.columns and 'is_weekend' in df_advanced.columns:
        df_advanced['temp_weekend_interaction'] = df_advanced['temperature'] * df_advanced['is_weekend']
    
    # 3. 统计特征（趋势和加速度）
    if 'orders_total' in df_advanced.columns:
        df_advanced['order_trend'] = df_advanced['orders_total'].diff(periods=1)
        df_advanced['order_acceleration'] = df_advanced['order_trend'].diff(periods=1)
        
        # 订单变化率
        df_advanced['order_change_rate'] = df_advanced['orders_total'].pct_change(periods=1)
    
    # 4. 分箱特征
    if 'hour' in df_advanced.columns:
        df_advanced['hour_bin'] = pd.cut(
            df_advanced['hour'], 
            bins=[0, 6, 11, 14, 18, 23], 
            labels=['深夜', '早晨', '午前', '午后', '夜晚'],
            include_lowest=True
        )
        # One-hot编码
        hour_bin_dummies = pd.get_dummies(df_advanced['hour_bin'], prefix='hour_bin')
        df_advanced = pd.concat([df_advanced, hour_bin_dummies], axis=1)
        df_advanced = df_advanced.drop('hour_bin', axis=1)
    
    if 'temperature' in df_advanced.columns:
        df_advanced['temp_bin'] = pd.cut(
            df_advanced['temperature'],
            bins=[0, 18, 25, 30, 40],
            labels=['寒冷', '舒适', '温暖', '炎热'],
            include_lowest=True
        )
        temp_bin_dummies = pd.get_dummies(df_advanced['temp_bin'], prefix='temp_bin')
        df_advanced = pd.concat([df_advanced, temp_bin_dummies], axis=1)
        df_advanced = df_advanced.drop('temp_bin', axis=1)
    
    logger.info(f"高级特征创建完成，当前特征数: {df_advanced.shape[1]}")
    return df_advanced

def extract_fourier_features(series, period, n_harmonics=3):
    """
    提取傅里叶特征
    
    参数:
        series: 时间序列数据
        period: 周期（如24小时、7天）
        n_harmonics: 谐波数量
    
    返回:
        DataFrame包含傅里叶特征
    """
    t = np.arange(len(series))
    features = pd.DataFrame()
    
    for n in range(1, n_harmonics + 1):
        features[f'fourier_sin_{period}_{n}'] = np.sin(2 * np.pi * n * t / period)
        features[f'fourier_cos_{period}_{n}'] = np.cos(2 * np.pi * n * t / period)
    
    return features

def create_region_features(df):
    """
    创建区域相关特征
    
    参数:
        df: 包含各区域订单量的DataFrame
    
    返回:
        添加了区域特征的DataFrame
    """
    logger.info("创建区域特征...")
    df_region = df.copy()
    
    region_cols = [col for col in df.columns if col.startswith('region_') and col.endswith('_orders')]
    
    if region_cols:
        # 区域订单占比
        for col in region_cols:
            region_name = col.replace('_orders', '_ratio')
            df_region[region_name] = df_region[col] / (df_region['orders_total'] + 1e-6)
        
        # 区域订单方差（衡量区域不平衡程度）
        df_region['region_variance'] = df_region[region_cols].var(axis=1)
        
        # 最繁忙区域
        df_region['busiest_region'] = df_region[region_cols].idxmax(axis=1)
        
        # 区域集中度（基尼系数简化版）
        region_orders = df_region[region_cols].values
        df_region['region_concentration'] = (region_orders.max(axis=1) - region_orders.min(axis=1)) / (df_region['orders_total'] + 1e-6)
    
    logger.info(f"区域特征创建完成")
    return df_region

if __name__ == "__main__":
    # 测试特征工程模块
    import sys
    sys.path.append('..')
    from data.data_generator import generate_simulated_data
    from src.data_preprocessing import DataPreprocessor
    
    logging.basicConfig(level=logging.INFO)
    
    df = generate_simulated_data(days=7)
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_all_features(df_clean)
    df_advanced = create_advanced_features(df_features)
    df_final = create_region_features(df_advanced)
    
    print(f"\n最终特征数: {df_final.shape[1]}")
    print(f"\n新增特征示例:\n{df_final[['hour_sin', 'hour_cos', 'temp_weather_interaction']].head()}")
