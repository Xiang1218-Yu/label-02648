"""
外卖订单预测与调度优化系统 - 主程序入口
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 导入项目模块
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import Config
    if not hasattr(Config, 'LOG_LEVEL'):
        raise ImportError("Config missing expected attributes")
except Exception as e:
    print(f"Falling back to root config.py due to: {e}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", os.path.join(os.path.dirname(__file__), 'config.py'))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    Config = config_module.Config

from data.data_generator import generate_simulated_data
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import create_advanced_features, create_region_features
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.scheduling_optimization import SchedulingOptimizer
from src.eda import EDAAnalyzer

# 配置日志
def setup_logging():
    """配置日志系统"""
    Config.create_directories()
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(os.path.join(Config.LOGS_DIR, 'project.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# 配置matplotlib中文显示
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass  # 如果字体配置失败，使用默认设置

def main():
    """主函数 - 执行整个项目流程"""
    logger = setup_logging()
    
    print("\n" + "="*70)
    print("外卖订单量预测与调度优化系统".center(70))
    print("="*70)
    
    # ========== 1. 生成/加载数据 ==========
    print("\n【步骤 1/7】 数据加载...")
    logger.info("="*50)
    logger.info("步骤 1: 数据加载")
    logger.info("="*50)
    
    if not os.path.exists(Config.DATA_PATH):
        logger.info(f"数据文件不存在，生成模拟数据（{Config.SIMULATION_DAYS}天）...")
        df = generate_simulated_data(days=Config.SIMULATION_DAYS)
        df.to_csv(Config.DATA_PATH, index=False)
        logger.info(f"数据已保存: {Config.DATA_PATH}")
    else:
        logger.info(f"加载已有数据: {Config.DATA_PATH}")
        df = pd.read_csv(Config.DATA_PATH)
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
    
    # ========== 1.5. 探索性数据分析（可选） ==========
    if Config.ENABLE_EDA:
        print("【可选步骤】 探索性数据分析...")
        logger.info("="*50)
        logger.info("可选步骤: 探索性数据分析 (EDA)")
        logger.info("="*50)
        
        eda_analyzer = EDAAnalyzer()
        eda_analyzer.df = df
        eda_analyzer.generate_full_report(
            output_dir=os.path.join(Config.RESULTS_DIR, 'eda')
        )
        logger.info("EDA分析完成")
    
    # ========== 2. 数据预处理 ==========
    print("【步骤 2/7】 数据预处理...")
    logger.info("="*50)
    logger.info("步骤 2: 数据预处理")
    logger.info("="*50)
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_all_features(
        df_clean, 
        lags=Config.LAG_FEATURES, 
        windows=Config.ROLLING_WINDOWS
    )
    
    logger.info(f"预处理后数据形状: {df_features.shape}")
    
    # ========== 3. 特征工程 ==========
    print("【步骤 3/7】 特征工程...")
    logger.info("="*50)
    logger.info("步骤 3: 特征工程")
    logger.info("="*50)
    
    df_advanced = create_advanced_features(df_features)
    df_final = create_region_features(df_advanced)
    
    logger.info(f"特征工程后数据形状: {df_final.shape}")
    logger.info(f"总特征数: {df_final.shape[1]}")
    
    # ========== 4. 准备训练数据 ==========
    print("【步骤 4/7】 准备训练数据...")
    logger.info("="*50)
    logger.info("步骤 4: 准备训练数据")
    logger.info("="*50)
    
    # 选择特征列（排除目标变量、时间戳和非数值列）
    exclude_cols = ['datetime', 'orders_total', 'busiest_region']
    region_order_cols = [col for col in df_final.columns 
                        if col.startswith('region_') and col.endswith('_orders')]
    exclude_cols.extend(region_order_cols)
    
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]
    
    X = df_final[feature_cols]
    y = df_final['orders_total']
    
    logger.info(f"特征数量: {X.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    
    # 划分训练集和测试集（时间序列不打乱）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, 
        shuffle=False
    )
    
    logger.info(f"训练集大小: {X_train.shape[0]}")
    logger.info(f"测试集大小: {X_test.shape[0]}")
    
    # ========== 5. 训练模型 ==========
    print("【步骤 5/7】 模型训练...")
    logger.info("="*50)
    logger.info("步骤 5: 模型训练")
    logger.info("="*50)
    
    trainer = ModelTrainer(random_state=Config.RANDOM_STATE)
    models = trainer.train_all_models(
        X_train, y_train, 
        models_to_train=Config.MODELS_TO_TRAIN,
        use_grid_search=False  # 快速训练，如需最优参数可设为True
    )
    
    # 保存模型
    trainer.save_models(Config.MODELS_DIR)
    
    # ========== 6. 评估模型 ==========
    print("【步骤 6/7】 模型评估...")
    logger.info("="*50)
    logger.info("步骤 6: 模型评估")
    logger.info("="*50)
    
    evaluator = ModelEvaluator(models)
    metrics = evaluator.calculate_metrics(X_test, y_test)
    
    # 生成可视化图表
    evaluator.plot_predictions(
        X_test, y_test, 
        sample_size=200,
        save_path=os.path.join(Config.RESULTS_DIR, 'predictions_comparison.png')
    )
    
    evaluator.plot_feature_importance(
        X.columns,
        top_n=15,
        save_path=os.path.join(Config.RESULTS_DIR, 'feature_importance.png')
    )
    
    evaluator.plot_time_series_comparison(
        y_test.values,
        save_path=os.path.join(Config.RESULTS_DIR, 'time_series_comparison.png')
    )
    
    # 导出性能指标
    metrics_df = evaluator.export_metrics_to_excel(
        os.path.join(Config.RESULTS_DIR, 'performance_metrics.xlsx')
    )
    
    # 打印评估结果
    print("\n" + "="*70)
    print("模型性能评估结果".center(70))
    print("="*70)
    print(metrics_df.to_string())
    print("="*70)
    
    # ========== 7. 生成调度建议 ==========
    print("\n【步骤 7/7】 生成调度建议...")
    logger.info("="*50)
    logger.info("步骤 7: 生成调度建议")
    logger.info("="*50)
    
    # 选择最佳模型
    best_model_name = max(metrics, key=lambda x: metrics[x]['R2'])
    best_model = models[best_model_name]
    
    logger.info(f"最佳模型: {best_model_name} (R²={metrics[best_model_name]['R2']:.4f})")
    
    # 使用测试集进行预测（模拟未来24小时）
    future_predictions = best_model.predict(X_test[:24])
    
    # 获取区域预测
    region_predictions = df_final.iloc[-24:][region_order_cols].values
    
    # 生成调度建议
    optimizer = SchedulingOptimizer(
        rider_efficiency=Config.RIDER_EFFICIENCY,
        max_riders_per_region=Config.MAX_RIDERS_PER_REGION
    )
    
    recommendations = optimizer.generate_schedule_recommendations(
        predictions=future_predictions,
        weather_forecast=1,  # 假设多云天气
        region_predictions=region_predictions
    )
    
    # 保存调度建议
    optimizer.save_recommendations(
        recommendations,
        os.path.join(Config.RESULTS_DIR, 'schedule_recommendations.json')
    )
    
    # 打印调度建议
    optimizer.print_recommendations(recommendations)
    
    # ========== 8. 保存预测结果 ==========
    logger.info("保存预测结果...")
    
    predictions_df = pd.DataFrame({
        'actual_orders': y_test.values,
        f'{best_model_name}_prediction': best_model.predict(X_test)
    })
    
    for model_name, model in models.items():
        if model_name != best_model_name:
            predictions_df[f'{model_name}_prediction'] = model.predict(X_test)
    
    predictions_df.to_csv(
        os.path.join(Config.RESULTS_DIR, 'predictions.csv'),
        index=False
    )
    logger.info(f"预测结果已保存: {os.path.join(Config.RESULTS_DIR, 'predictions.csv')}")
    
    # ========== 完成 ==========
    print("\n" + "="*70)
    print("项目执行完成".center(70))
    print("="*70)
    print(f"\n✓ 最佳模型: {best_model_name}")
    print(f"✓ R² 分数: {metrics[best_model_name]['R2']:.4f}")
    print(f"✓ RMSE: {metrics[best_model_name]['RMSE']:.2f}")
    print(f"✓ MAPE: {metrics[best_model_name]['MAPE']:.2f}%")
    print(f"\n✓ 模型文件: {Config.MODELS_DIR}/")
    print(f"✓ 结果文件: {Config.RESULTS_DIR}/")
    print(f"✓ 日志文件: {Config.LOGS_DIR}/project.log")
    print("="*70 + "\n")
    
    return models, metrics, recommendations

if __name__ == "__main__":
    try:
        models, metrics, recommendations = main()
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}", exc_info=True)
        print(f"\n❌ 错误: {str(e)}")
        sys.exit(1)
