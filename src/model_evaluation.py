"""
模型评估模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             r2_score, explained_variance_score)
import logging
import os

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, models):
        self.models = models
        self.metrics = {}
        self.predictions = {}
        
    def calculate_mape(self, y_true, y_pred):
        """计算平均绝对百分比误差"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # 避免除零
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_metrics(self, X_test, y_test):
        """计算各项评估指标"""
        logger.info("开始计算评估指标...")
        
        for name, model in self.models.items():
            logger.info(f"评估模型: {name}")
            y_pred = model.predict(X_test)
            self.predictions[name] = y_pred
            
            # 计算回归评估指标
            metrics = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred),
                'MAPE': self.calculate_mape(y_test, y_pred),
                'Explained_Variance': explained_variance_score(y_test, y_pred)
            }
            
            self.metrics[name] = metrics
            
            # 打印指标
            logger.info(f"{name} 评估指标:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        
        return self.metrics
    
    def plot_predictions(self, X_test, y_test, sample_size=100, save_path=None):
        """绘制预测结果对比图"""
        logger.info("绘制预测对比图...")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = self.predictions.get(name, model.predict(X_test))
            
            # 预测vs实际散点图
            axes[0, idx].scatter(y_test[:sample_size], y_pred[:sample_size], 
                               alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
            axes[0, idx].plot([y_test.min(), y_test.max()], 
                            [y_test.min(), y_test.max()], 
                            'r--', lw=2, label='理想预测线')
            axes[0, idx].set_xlabel('实际订单量', fontsize=12)
            axes[0, idx].set_ylabel('预测订单量', fontsize=12)
            axes[0, idx].set_title(f'{name} - 预测 vs 实际\nR²={self.metrics[name]["R2"]:.4f}', 
                                  fontsize=14, fontweight='bold')
            axes[0, idx].legend()
            axes[0, idx].grid(True, alpha=0.3)
            
            # 残差图
            residuals = y_test - y_pred
            axes[1, idx].scatter(y_pred[:sample_size], residuals[:sample_size], 
                               alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
            axes[1, idx].axhline(y=0, color='r', linestyle='--', lw=2, label='零残差线')
            axes[1, idx].set_xlabel('预测订单量', fontsize=12)
            axes[1, idx].set_ylabel('残差', fontsize=12)
            axes[1, idx].set_title(f'{name} - 残差分析\nMAE={self.metrics[name]["MAE"]:.2f}', 
                                  fontsize=14, fontweight='bold')
            axes[1, idx].legend()
            axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"预测对比图已保存: {save_path}")
        
        plt.close()
        return fig
    
    def plot_feature_importance(self, feature_names, top_n=15, save_path=None):
        """绘制特征重要性图"""
        logger.info("绘制特征重要性图...")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, model) in enumerate(self.models.items()):
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n]
                
                # 绘制水平条形图
                y_pos = np.arange(len(indices))
                axes[idx].barh(y_pos, importances[indices], align='center', alpha=0.8)
                axes[idx].set_yticks(y_pos)
                axes[idx].set_yticklabels([feature_names[i] for i in indices], fontsize=10)
                axes[idx].invert_yaxis()
                axes[idx].set_xlabel('特征重要性', fontsize=12)
                axes[idx].set_title(f'{name} - Top {top_n} 特征重要性', 
                                   fontsize=14, fontweight='bold')
                axes[idx].grid(True, alpha=0.3, axis='x')
            else:
                axes[idx].text(0.5, 0.5, f'{name}\n不支持特征重要性分析', 
                             ha='center', va='center', fontsize=14)
                axes[idx].set_xlim(0, 1)
                axes[idx].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"特征重要性图已保存: {save_path}")
        
        plt.close()
        return fig
    
    def plot_time_series_comparison(self, y_test, timestamps=None, save_path=None):
        """绘制时间序列预测对比"""
        logger.info("绘制时间序列对比图...")
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        if timestamps is None:
            timestamps = np.arange(len(y_test))
        
        # 绘制实际值
        ax.plot(timestamps, y_test, label='实际订单量', 
               linewidth=2, color='black', alpha=0.7)
        
        # 绘制各模型预测值
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, (name, y_pred) in enumerate(self.predictions.items()):
            ax.plot(timestamps, y_pred, label=f'{name}预测', 
                   linewidth=1.5, alpha=0.7, color=colors[idx % len(colors)])
        
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('订单量', fontsize=12)
        ax.set_title('时间序列预测对比', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"时间序列对比图已保存: {save_path}")
        
        plt.close()
        return fig
    
    def export_metrics_to_excel(self, filepath='results/performance_metrics.xlsx'):
        """导出性能指标到Excel"""
        logger.info(f"导出性能指标到: {filepath}")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 转换为DataFrame
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.index.name = '模型名称'
        
        # 保存到Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='性能指标')
        
        logger.info("性能指标导出完成")
        return metrics_df

if __name__ == "__main__":
    # 测试评估模块
    import sys
    sys.path.append('..')
    from data.data_generator import generate_simulated_data
    from src.data_preprocessing import DataPreprocessor
    from src.feature_engineering import create_advanced_features
    from src.model_training import ModelTrainer
    from sklearn.model_selection import train_test_split
    
    logging.basicConfig(level=logging.INFO)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据和模型
    df = generate_simulated_data(days=30)
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_all_features(df_clean)
    df_advanced = create_advanced_features(df_features)
    
    feature_cols = [col for col in df_advanced.columns 
                    if col not in ['datetime', 'orders_total'] and 
                    not col.startswith('region_') and
                    not col.startswith('busiest_')]
    
    X = df_advanced[feature_cols]
    y = df_advanced['orders_total']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train, y_train, use_grid_search=False)
    
    # 评估模型
    evaluator = ModelEvaluator(models)
    metrics = evaluator.calculate_metrics(X_test, y_test)
    evaluator.plot_predictions(X_test, y_test)
    evaluator.plot_feature_importance(X.columns)
    
    print("\n评估完成")
