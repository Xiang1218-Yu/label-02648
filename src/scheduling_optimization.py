"""
调度优化模块
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import logging
import json
import os
from .utils import convert_to_json_serializable

logger = logging.getLogger(__name__)

class SchedulingOptimizer:
    def __init__(self, rider_efficiency=10, max_riders_per_region=50):
        """
        初始化调度优化器
        
        参数:
            rider_efficiency: 每个骑手每小时处理订单数，默认10单/小时
            max_riders_per_region: 每个区域最大骑手数
        """
        self.rider_efficiency = rider_efficiency
        self.max_riders_per_region = max_riders_per_region
        self.region_centers = None
        
    def identify_hotspots(self, region_orders, n_clusters=3):
        """
        识别订单热点区域
        
        参数:
            region_orders: 各区域订单量数组 (n_samples, n_regions)
            n_clusters: 聚类数量
        
        返回:
            clusters: 聚类标签
        """
        logger.info(f"识别订单热点区域，聚类数: {n_clusters}")
        
        # 使用K-Means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(region_orders)
        self.region_centers = kmeans.cluster_centers_
        
        logger.info(f"热点识别完成，聚类中心:\n{self.region_centers}")
        return clusters
    
    def calculate_rider_allocation(self, predicted_orders, current_riders=None):
        """
        计算骑手分配方案
        
        参数:
            predicted_orders: 预测订单量 (可以是单个值或数组)
            current_riders: 当前各区域骑手数量
        
        返回:
            required_riders: 各区域建议骑手数量
        """
        logger.info("计算骑手分配方案...")
        
        # 确保输入是数组
        if isinstance(predicted_orders, (int, float)):
            predicted_orders = np.array([predicted_orders])
        else:
            predicted_orders = np.array(predicted_orders)
        
        # 计算所需骑手数（向上取整）
        required_riders = np.ceil(predicted_orders / self.rider_efficiency).astype(int)
        
        # 限制最大骑手数
        required_riders = np.minimum(required_riders, self.max_riders_per_region)
        
        # 如果提供了当前骑手数，进行优化分配
        if current_riders is not None:
            current_riders = np.array(current_riders)
            allocation = self.optimize_allocation(required_riders, current_riders)
            return allocation
        
        return required_riders
    
    def optimize_allocation(self, required, current):
        """
        优化骑手分配算法（贪心算法）
        
        参数:
            required: 各区域需求骑手数
            current: 各区域当前骑手数
        
        返回:
            allocation: 优化后的骑手分配
        """
        logger.info("优化骑手分配...")
        
        allocation = current.copy()
        
        # 识别短缺和富余区域
        shortage_regions = np.where(required > current)[0]
        surplus_regions = np.where(required < current)[0]
        
        logger.info(f"短缺区域: {shortage_regions}, 富余区域: {surplus_regions}")
        
        # 从富余区域调配到短缺区域
        for shortage_idx in shortage_regions:
            needed = required[shortage_idx] - allocation[shortage_idx]
            
            for surplus_idx in surplus_regions:
                available = allocation[surplus_idx] - required[surplus_idx]
                
                if available > 0 and needed > 0:
                    transfer = min(needed, available)
                    allocation[shortage_idx] += transfer
                    allocation[surplus_idx] -= transfer
                    needed -= transfer
                    
                    logger.info(f"从区域{surplus_idx}调配{transfer}名骑手到区域{shortage_idx}")
            
            if needed > 0:
                logger.warning(f"区域{shortage_idx}仍缺少{needed}名骑手")
        
        return allocation
    
    def generate_schedule_recommendations(self, predictions, weather_forecast=None, 
                                         region_predictions=None):
        """
        生成调度建议报告
        
        参数:
            predictions: 总订单量预测 (数组)
            weather_forecast: 天气预报 (0-4)
            region_predictions: 各区域订单预测 (DataFrame或数组)
        
        返回:
            recommendations: 调度建议字典
        """
        logger.info("生成调度建议报告...")
        
        recommendations = {
            'peak_hours': [],
            'weather_adjustments': [],
            'region_priorities': [],
            'total_riders_needed': 0,
            'hourly_allocation': []
        }
        
        # 确保predictions是数组
        predictions = np.array(predictions)
        
        # 1. 分析高峰时段（订单量超过75分位数）
        threshold = np.percentile(predictions, 75)
        peak_indices = np.where(predictions > threshold)[0]
        peak_hours = [int(x) for x in list(set(peak_indices % 24))]  # 确保转换为 int
        recommendations['peak_hours'] = sorted(peak_hours)
        
        logger.info(f"识别高峰时段: {recommendations['peak_hours']}")
        
        # 2. 天气调整建议
        if weather_forecast is not None:
            weather_adjustments = {
                0: {'type': 'normal', 'percentage': 0, 'reason': '晴天，正常调度'},
                1: {'type': 'normal', 'percentage': -0.05, 'reason': '多云，订单量可能略微下降5%'},
                2: {'type': 'reduce', 'percentage': -0.15, 'reason': '小雨，订单量预计下降15%'},
                3: {'type': 'reduce', 'percentage': -0.30, 'reason': '大雨，订单量预计下降30%'},
                4: {'type': 'reduce', 'percentage': -0.50, 'reason': '极端天气，订单量预计下降50%'}
            }
            
            if weather_forecast in weather_adjustments:
                recommendations['weather_adjustments'].append(
                    weather_adjustments[weather_forecast]
                )
        
        # 3. 区域优先级分析
        if region_predictions is not None:
            if isinstance(region_predictions, pd.DataFrame):
                region_cols = [col for col in region_predictions.columns 
                              if col.startswith('region_') and col.endswith('_orders')]
                region_totals = region_predictions[region_cols].sum()
            else:
                region_totals = np.sum(region_predictions, axis=0)
                region_cols = [f'region_{i}' for i in range(len(region_totals))]
            
            # 按订单量排序
            sorted_regions = sorted(zip(region_cols, region_totals), 
                                   key=lambda x: x[1], reverse=True)
            
            recommendations['region_priorities'] = [
                {'region': region, 'total_orders': int(orders), 
                 'priority': idx + 1}
                for idx, (region, orders) in enumerate(sorted_regions)
            ]
        
        # 4. 计算总骑手需求
        total_riders = self.calculate_rider_allocation(predictions)
        recommendations['total_riders_needed'] = int(np.sum(total_riders))
        
        # 5. 每小时分配建议
        for hour_idx, (pred, riders) in enumerate(zip(predictions[:24], total_riders[:24])):
            recommendations['hourly_allocation'].append({
                'hour': hour_idx,
                'predicted_orders': int(pred),
                'required_riders': int(riders),
                'utilization_rate': round(pred / (riders * self.rider_efficiency) * 100, 2) if riders > 0 else 0
            })
        
        logger.info(f"调度建议生成完成，总需求骑手: {recommendations['total_riders_needed']}")
        return recommendations
    
    def save_recommendations(self, recommendations, filepath='results/schedule_recommendations.json'):
        """保存调度建议到JSON文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 转换 numpy 类型为 Python 原生类型
        recommendations_native = convert_to_json_serializable(recommendations)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(recommendations_native, f, ensure_ascii=False, indent=2)
        
        logger.info(f"调度建议已保存: {filepath}")
    
    def print_recommendations(self, recommendations):
        """打印调度建议摘要"""
        print("\n" + "="*60)
        print("调度优化建议报告".center(60))
        print("="*60)
        
        print(f"\n【高峰时段】")
        print(f"  预测高峰时段: {recommendations['peak_hours']}点")
        
        print(f"\n【天气影响】")
        for adj in recommendations['weather_adjustments']:
            print(f"  {adj['reason']}")
        
        print(f"\n【区域优先级】")
        for region_info in recommendations['region_priorities'][:5]:
            print(f"  优先级{region_info['priority']}: {region_info['region']} "
                  f"(预计{region_info['total_orders']}单)")
        
        print(f"\n【骑手需求】")
        print(f"  总需求骑手数: {recommendations['total_riders_needed']}人")
        
        print(f"\n【高峰时段分配】")
        peak_allocations = sorted(recommendations['hourly_allocation'], 
                                 key=lambda x: x['predicted_orders'], reverse=True)[:5]
        for alloc in peak_allocations:
            print(f"  {alloc['hour']}点: 预计{alloc['predicted_orders']}单, "
                  f"需要{alloc['required_riders']}名骑手 "
                  f"(利用率{alloc['utilization_rate']}%)")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    # 测试调度优化模块
    logging.basicConfig(level=logging.INFO)
    
    # 模拟预测数据
    predictions = np.random.poisson(150, 24)  # 24小时预测
    region_predictions = np.random.poisson(30, (24, 5))  # 5个区域
    
    optimizer = SchedulingOptimizer(rider_efficiency=10)
    
    # 生成调度建议
    recommendations = optimizer.generate_schedule_recommendations(
        predictions, 
        weather_forecast=1,
        region_predictions=region_predictions
    )
    
    # 打印建议
    optimizer.print_recommendations(recommendations)
    
    print("\n调度优化测试完成")
