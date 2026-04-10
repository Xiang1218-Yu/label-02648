#!/usr/bin/env python
"""
Tests for the scheduling_optimization module.
"""
import pytest
import numpy as np
import pandas as pd
import json
import os
import tempfile

from src.scheduling_optimization import SchedulingOptimizer


class TestSchedulingOptimizer:
    def test_initialization_default(self):
        optimizer = SchedulingOptimizer()
        assert optimizer.rider_efficiency == 10
        assert optimizer.max_riders_per_region == 50
        assert optimizer.region_centers is None

    def test_initialization_custom(self):
        optimizer = SchedulingOptimizer(rider_efficiency=15, max_riders_per_region=100)
        assert optimizer.rider_efficiency == 15
        assert optimizer.max_riders_per_region == 100

    def test_identify_hotspots_basic(self):
        optimizer = SchedulingOptimizer()
        region_orders = np.array([[10, 20], [30, 40], [15, 25], [35, 45]])
        clusters = optimizer.identify_hotspots(region_orders, n_clusters=2)
        
        assert len(clusters) == 4
        assert optimizer.region_centers is not None
        assert optimizer.region_centers.shape == (2, 2)

    def test_identify_hotspots_single_cluster(self):
        optimizer = SchedulingOptimizer()
        region_orders = np.array([[10, 20], [15, 25], [12, 22]])
        clusters = optimizer.identify_hotspots(region_orders, n_clusters=1)
        
        assert len(clusters) == 3
        assert all(c == 0 for c in clusters)

    def test_calculate_rider_allocation_single_value(self):
        optimizer = SchedulingOptimizer(rider_efficiency=10)
        result = optimizer.calculate_rider_allocation(150)
        
        assert result.shape == (1,)
        assert result[0] == 15  # ceil(150/10) = 15

    def test_calculate_rider_allocation_array(self):
        optimizer = SchedulingOptimizer(rider_efficiency=10)
        predictions = [15, 25, 8, 105]
        result = optimizer.calculate_rider_allocation(predictions)
        
        assert len(result) == 4
        assert list(result) == [2, 3, 1, 11]  # ceil(15/10)=2, ceil(25/10)=3, etc.

    def test_calculate_rider_allocation_max_limit(self):
        optimizer = SchedulingOptimizer(rider_efficiency=10, max_riders_per_region=50)
        result = optimizer.calculate_rider_allocation(1000)  # Would need 100 riders
        
        assert result[0] == 50  # Limited by max_riders_per_region

    def test_optimize_allocation_no_transfer_needed(self):
        optimizer = SchedulingOptimizer()
        required = np.array([5, 10, 3])
        current = np.array([5, 10, 3])
        
        allocation = optimizer.optimize_allocation(required, current)
        assert list(allocation) == [5, 10, 3]

    def test_optimize_allocation_with_transfer(self):
        optimizer = SchedulingOptimizer()
        required = np.array([10, 5, 3])
        current = np.array([5, 10, 3])
        
        allocation = optimizer.optimize_allocation(required, current)
        
        # Should transfer riders from region 1 to region 0
        assert allocation[0] >= 5
        assert allocation[1] <= 10
        assert sum(allocation) == sum(current)  # Total riders remain the same

    def test_optimize_allocation_shortage_warning(self, caplog):
        optimizer = SchedulingOptimizer()
        required = np.array([20, 5])
        current = np.array([5, 5])  # Not enough total riders
        
        allocation = optimizer.optimize_allocation(required, current)
        
        assert allocation[0] == 5  # Cannot get riders: region 1 allocation equals required (=5)
        assert allocation[1] == 5

    def test_generate_schedule_recommendations_basic(self):
        optimizer = SchedulingOptimizer()
        predictions = np.array([100, 150, 80, 200] * 6)  # 24 hours worth
        
        recommendations = optimizer.generate_schedule_recommendations(predictions)
        
        assert 'peak_hours' in recommendations
        assert 'weather_adjustments' in recommendations
        assert 'region_priorities' in recommendations
        assert 'total_riders_needed' in recommendations
        assert 'hourly_allocation' in recommendations
        
        assert recommendations['total_riders_needed'] > 0
        assert len(recommendations['hourly_allocation']) == min(24, len(predictions))  # min 24 hours

    def test_generate_schedule_recommendations_weather(self):
        optimizer = SchedulingOptimizer()
        predictions = np.random.poisson(100, 24)
        
        recommendations = optimizer.generate_schedule_recommendations(
            predictions, weather_forecast=2
        )
        
        assert len(recommendations['weather_adjustments']) == 1
        adj = recommendations['weather_adjustments'][0]
        assert adj['type'] == 'reduce'
        assert adj['percentage'] == -0.15

    def test_generate_schedule_recommendations_region_priorities_with_df(self):
        optimizer = SchedulingOptimizer()
        predictions = np.random.poisson(100, 24)
        region_df = pd.DataFrame({
            'region_0_orders': [100, 150],
            'region_1_orders': [200, 250],
            'other_col': [1, 2]
        })
        
        recommendations = optimizer.generate_schedule_recommendations(
            predictions, region_predictions=region_df
        )
        
        assert len(recommendations['region_priorities']) == 2
        # Higher order region should come first
        assert recommendations['region_priorities'][0]['region'] == 'region_1_orders'
        assert recommendations['region_priorities'][0]['priority'] == 1

    def test_generate_schedule_recommendations_region_priorities_with_array(self):
        optimizer = SchedulingOptimizer()
        predictions = np.random.poisson(100, 24)
        region_array = np.array([[100, 200], [150, 250]])
        
        recommendations = optimizer.generate_schedule_recommendations(
            predictions, region_predictions=region_array
        )
        
        assert len(recommendations['region_priorities']) == 2
        assert 'region_' in recommendations['region_priorities'][0]['region']

    def test_save_recommendations(self):
        optimizer = SchedulingOptimizer()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            recommendations = {
                'peak_hours': [8, 9, 12],
                'total_riders_needed': 45,
                'nested': {'value': np.int64(10)}  # Contains numpy type
            }
            
            optimizer.save_recommendations(recommendations, temp_path)
            
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded['peak_hours'] == [8, 9, 12]
            assert loaded['total_riders_needed'] == 45
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_recommendations_creates_directory(self):
        optimizer = SchedulingOptimizer()
        temp_dir = tempfile.mkdtemp()
        nested_path = os.path.join(temp_dir, 'subdir', 'recs.json')
        
        try:
            recommendations = {'test': 'data'}
            optimizer.save_recommendations(recommendations, nested_path)
            
            assert os.path.exists(nested_path)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_print_recommendations(self, capsys):
        optimizer = SchedulingOptimizer()
        
        recommendations = {
            'peak_hours': [8, 12, 18],
            'weather_adjustments': [{'reason': 'Test weather reason'}],
            'region_priorities': [
                {'priority': 1, 'region': 'region_a', 'total_orders': 1000},
                {'priority': 2, 'region': 'region_b', 'total_orders': 500}
            ],
            'total_riders_needed': 50,
            'hourly_allocation': [
                {'hour': 8, 'predicted_orders': 100, 'required_riders': 10, 'utilization_rate': 100.0},
                {'hour': 12, 'predicted_orders': 150, 'required_riders': 15, 'utilization_rate': 100.0}
            ]
        }
        
        optimizer.print_recommendations(recommendations)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert '高峰时段' in output
        assert '天气影响' in output
        assert '区域优先级' in output
        assert '骑手需求' in output
        assert '高峰时段分配' in output
        assert '50人' in output  # Total riders
        assert 'region_a' in output

    def test_recommendations_zero_riders_utilization(self):
        optimizer = SchedulingOptimizer(rider_efficiency=10)
        predictions = np.array([0, 0, 0])
        
        recommendations = optimizer.generate_schedule_recommendations(predictions)
        
        for alloc in recommendations['hourly_allocation']:
            assert alloc['utilization_rate'] == 0  # No division by zero when riders=0
