# 外卖订单预测与调度优化系统

## How to Run

### 环境要求
- Python 3.8+
- 操作系统：Windows / macOS / Linux
- 磁盘空间：至少 500MB

### 安装步骤

```bash
# 如果是python3 命令行改为python3
# 1. 创建虚拟环境（推荐）
python -m venv venv

# 2. 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. 安装依赖包
pip install -r requirements.txt

# 4. 运行项目
python main.py
```

### 运行说明
- **预计运行时间**：5-10分钟（取决于硬件配置）
- **输出内容**：
  - 生成模拟数据（90天，2160条记录）
  - 训练2个机器学习模型（随机森林、XGBoost）
  - 生成性能评估报告和可视化图表
  - 生成调度优化建议

### 查看结果

```bash
# 查看模型文件
ls models/

# 查看结果文件
ls results/

# 查看运行日志
cat logs/project.log
```

---

## Services

本项目是一个**本地运行的Python数据分析项目**，不涉及Web服务或API接口。

### 核心服务模块

| 模块名称 | 功能说明 | 输入 | 输出 |
|---------|---------|------|------|
| 数据生成服务 | 生成模拟外卖订单数据 | 天数参数 | CSV数据文件 |
| 数据预处理服务 | 数据清洗、特征工程 | 原始数据 | 特征数据集 |
| 模型训练服务 | 训练机器学习模型 | 特征数据 | 模型文件(.pkl) |
| 模型评估服务 | 评估模型性能 | 测试数据 | 性能指标、可视化图表 |
| 调度优化服务 | 生成骑手调度建议 | 预测结果 | 调度建议JSON |

### 技术架构

```
用户
  ↓
main.py (主程序入口)
  ↓
├─→ data_generator.py (数据生成)
├─→ data_preprocessing.py (数据预处理)
├─→ feature_engineering.py (特征工程)
├─→ model_training.py (模型训练)
├─→ model_evaluation.py (模型评估)
└─→ scheduling_optimization.py (调度优化)
  ↓
输出文件 (models/, results/, logs/)
```

---

## 测试账号

本项目为**数据分析项目**，无需登录账号。所有功能通过命令行运行。

### 测试数据说明

项目会自动生成模拟数据，包含以下特征：
- **时间范围**：90天（可在config.py中调整）
- **数据量**：2160条记录（90天 × 24小时）
- **订单量范围**：50-400单/小时
- **区域数量**：5个配送区域（A-E）
- **环境因素**：温度、天气、节假日等

---

## 题目内容
```
一、代码整体结构 
python 
# 项目整体结构 
外卖订单预测项目/ 
├── data/ 
│   ├── simulated_data.csv       # 模拟数据集 
│   └── data_generator.py        # 数据生成模块 
├── src/ 
│   ├── data_preprocessing.py    # 数据预处理模块 
│   ├── eda.py                   # 探索性数据分析模块 
│   ├── feature_engineering.py   # 特征工程模块 
│   ├── model_training.py        # 模型训练模块 
│   ├── model_evaluation.py      # 模型评估模块 
│   └── scheduling_optimization.py # 调度优化模块 
├── models/ 
│   ├── random_forest_model.pkl  # 随机森林模型 
│   └── xgboost_model.pkl        # XGBoost模型 
├── results/ 
│   ├── predictions.csv          # 预测结果 
│   └── performance_metrics.xlsx # 性能指标 
├── notebooks/ 
│   └── 外卖订单预测分析.ipynb    # Jupyter Notebook分析文件 
├── config.py                    # 配置文件 
├── main.py                      # 主程序入口 
└── requirements.txt             # 依赖包列表 
二、核心代码模块详细说明 
1. 数据生成模块 ( data_generator.py ) 
python 
def generate_simulated_data(days=90): 
""" 
生成模拟外卖订单数据 

参数: 
days: 模拟天数，默认90天 

返回: 
pandas DataFrame，包含以下列: 
- datetime: 时间戳 
- hour, weekday, is_weekend, is_holiday: 时间特征 
- temperature, weather: 环境特征 
- orders_total: 订单总量 
- region_A-E_orders: 各区域订单量 
""" 
# 核心逻辑: 
# 1. 创建时间序列（按小时） 
# 2. 生成每日/每周周期模式 
# 3. 添加外部影响因素（天气、温度、节假日） 
# 4. 加入随机噪声使数据更真实 
# 5. 生成各区域订单分布 
关键算法 ： 
使用正弦函数模拟每日双峰模式 
使用泊松分布模拟订单量的随机性 
使用狄利克雷分布模拟区域订单分配 
2. 数据预处理模块 ( data_preprocessing.py ) 
python 
class DataPreprocessor: 
def __init__(self): 
self.scaler = StandardScaler() 

def load_data(self, filepath): 
"""加载数据并检查缺失值""" 

def clean_data(self, df): 
"""数据清洗：处理缺失值、异常值""" 

def create_time_features(self, df): 
"""从datetime创建时间特征""" 
# 生成：year, month, day, hour, weekday, is_weekend, is_holiday 

def create_lag_features(self, df, lags=[1, 2, 3, 6, 12]): 
"""创建滞后特征""" 
# orders_lag_1h, orders_lag_2h, ... 

def create_rolling_features(self, df, windows=[3, 6, 12, 24]): 
"""创建滑动窗口特征""" 
# orders_rolling_mean_3h, orders_rolling_std_6h, ... 

def encode_categorical(self, df): 
"""对分类变量进行编码""" 
# one-hot编码天气、节假日等 

def normalize_features(self, df, features): 
"""特征标准化""" 
3. 特征工程模块 ( feature_engineering.py ) 
python 
def create_advanced_features(df): 
"""创建高级特征""" 

# 1. 周期性特征 
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24) 
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24) 
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7) 

# 2. 交互特征 
df['temp_weather_interaction'] = df['temperature'] * df['weather'] 
df['hour_weekend_interaction'] = df['hour'] * df['is_weekend'] 

# 3. 统计特征 
df['order_trend'] = df['orders_total'].diff(periods=1) 
df['order_acceleration'] = df['order_trend'].diff(periods=1) 

# 4. 分箱特征 
df['hour_bin'] = pd.cut(df['hour'], 
bins=[0, 6, 11, 14, 18, 23], 
labels=['深夜', '早晨', '午前', '午后', '夜晚']) 

return df 
4. 模型训练模块 ( model_training.py ) 
python 
class ModelTrainer: 
def __init__(self): 
self.models = {} 
self.results = {} 

def train_random_forest(self, X_train, y_train): 
"""训练随机森林模型""" 

# 定义参数网格 
param_grid = { 
'n_estimators': [100, 200, 300], 
'max_depth': [10, 20, 30, None], 
'min_samples_split': [2, 5, 10], 
'min_samples_leaf': [1, 2, 4] 
} 

# 网格搜索 
grid_search = GridSearchCV( 
RandomForestRegressor(random_state=42), 
param_grid, 
cv=5, 
scoring='r2', 
n_jobs=-1, 
verbose=1 
) 

grid_search.fit(X_train, y_train) 
return grid_search.best_estimator_ 

def train_xgboost(self, X_train, y_train): 
"""训练XGBoost模型""" 

param_grid = { 
'n_estimators': [100, 200, 300], 
'max_depth': [3, 5, 7, 9], 
'learning_rate': [0.01, 0.05, 0.1, 0.2], 
'subsample': [0.8, 0.9, 1.0], 
'colsample_bytree': [0.8, 0.9, 1.0] 
} 

grid_search = GridSearchCV( 
xgb.XGBRegressor(random_state=42), 
param_grid, 
cv=5, 
scoring='r2', 
n_jobs=-1, 
verbose=1 
) 

grid_search.fit(X_train, y_train) 
return grid_search.best_estimator_ 

def train_lightgbm(self, X_train, y_train): 
"""训练LightGBM模型（备选）""" 

def train_all_models(self, X_train, y_train): 
"""训练所有模型""" 
self.models['random_forest'] = self.train_random_forest(X_train, y_train) 
self.models['xgboost'] = self.train_xgboost(X_train, y_train) 
# 可以继续添加其他模型 
5. 模型评估模块 ( model_evaluation.py ) 
python 
class ModelEvaluator: 
def __init__(self, models): 
self.models = models 
self.metrics = {} 

def calculate_metrics(self, X_test, y_test): 
"""计算各项评估指标""" 

for name, model in self.models.items(): 
y_pred = model.predict(X_test) 

# 回归评估指标 
metrics = { 
'MAE': mean_absolute_error(y_test, y_pred), 
'MSE': mean_squared_error(y_test, y_pred), 
'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)), 
'R2': r2_score(y_test, y_pred), 
'MAPE': self.calculate_mape(y_test, y_pred), 
'Explained Variance': explained_variance_score(y_test, y_pred) 
} 

self.metrics[name] = metrics 

return self.metrics 

def plot_predictions(self, X_test, y_test, sample_size=100): 
"""绘制预测结果对比图""" 
fig, axes = plt.subplots(2, 2, figsize=(15, 12)) 

for idx, (name, model) in enumerate(self.models.items()): 
y_pred = model.predict(X_test) 

# 预测vs实际散点图 
axes[0, idx].scatter(y_test[:sample_size], y_pred[:sample_size], alpha=0.5) 
axes[0, idx].plot([y_test.min(), y_test.max()], 
[y_test.min(), y_test.max()], 'r--', lw=2) 
axes[0, idx].set_title(f'{name}预测 vs 实际') 

# 残差图 
residuals = y_test - y_pred 
axes[1, idx].scatter(y_pred[:sample_size], residuals[:sample_size], alpha=0.5) 
axes[1, idx].axhline(y=0, color='r', linestyle='--') 
axes[1, idx].set_title(f'{name}残差图') 

def plot_feature_importance(self, feature_names): 
"""绘制特征重要性图""" 
fig, axes = plt.subplots(1, len(self.models), figsize=(15, 6)) 

for idx, (name, model) in enumerate(self.models.items()): 
if hasattr(model, 'feature_importances_'): 
importances = model.feature_importances_ 
indices = np.argsort(importances)[::-1][:10]  # 取前10个 

axes[idx].barh(range(10), importances[indices]) 
axes[idx].set_yticks(range(10)) 
axes[idx].set_yticklabels([feature_names[i] for i in indices]) 
axes[idx].set_title(f'{name}特征重要性') 
6. 调度优化模块 ( scheduling_optimization.py ) 
python 
class SchedulingOptimizer: 
def __init__(self): 
self.region_centers = None 

def identify_hotspots(self, region_orders, n_clusters=3): 
"""识别订单热点区域""" 
# 使用K-Means聚类识别热点 
kmeans = KMeans(n_clusters=n_clusters, random_state=42) 
clusters = kmeans.fit_predict(region_orders) 
self.region_centers = kmeans.cluster_centers_ 

return clusters 

def calculate_rider_allocation(self, predicted_orders, 
current_riders, efficiency=10): 
""" 
计算骑手分配方案 

参数: 
predicted_orders: 预测订单量 
current_riders: 当前各区域骑手数量 
efficiency: 每个骑手每小时处理订单数，默认10单/小时 

返回: 
各区域建议骑手数量 
""" 
required_riders = np.ceil(predicted_orders / efficiency).astype(int) 

# 考虑骑手移动成本 
allocation = self.optimize_allocation(required_riders, current_riders) 

return allocation 

def optimize_allocation(self, required, current): 
"""优化分配算法（使用线性规划）""" 
from scipy.optimize import linprog 

# 目标函数：最小化总移动成本 
# 约束条件：每个区域骑手数 >= 需求数 

# 这里简化为贪心算法 
allocation = current.copy() 
shortage_regions = np.where(required > current)[0] 
surplus_regions = np.where(required < current)[0] 

# 从富余区域调配到短缺区域 
for shortage in shortage_regions: 
needed = required[shortage] - allocation[shortage] 
for surplus in surplus_regions: 
available = allocation[surplus] - required[surplus] 
if available > 0: 
transfer = min(needed, available) 
allocation[shortage] += transfer 
allocation[surplus] -= transfer 
needed -= transfer 

return allocation 

def generate_schedule_recommendations(self, predictions, weather_forecast): 
"""生成调度建议报告""" 
recommendations = { 
'peak_hours': [], 
'weather_adjustments': [], 
'region_priorities': [], 
'total_riders_needed': 0 
} 

# 分析高峰时段 
peak_indices = np.where(predictions > np.percentile(predictions, 75))[0] 
recommendations['peak_hours'] = list(set(peak_indices % 24)) 

# 天气调整建议 
if weather_forecast in [3, 4]:  # 恶劣天气 
recommendations['weather_adjustments'].append({ 
'type': 'reduce', 
'percentage': 0.4, 
'reason': '恶劣天气订单量预计下降40%' 
}) 

return recommendations 
7. 主程序入口 ( main.py ) 
python 
def main(): 
"""主函数 - 执行整个项目流程""" 

print("=== 外卖订单量预测与调度优化系统 ===") 

# 1. 生成/加载数据 
print("\n1. 加载数据...") 
if not os.path.exists('data/simulated_data.csv'): 
df = generate_simulated_data(90) 
df.to_csv('data/simulated_data.csv', index=False) 
else: 
df = pd.read_csv('data/simulated_data.csv') 

# 2. 数据预处理 
print("2. 数据预处理...") 
preprocessor = DataPreprocessor() 
df_clean = preprocessor.clean_data(df) 
df_features = preprocessor.create_all_features(df_clean) 

# 3. 划分数据集 
X = df_features.drop('orders_total', axis=1) 
y = df_features['orders_total'] 
X_train, X_test, y_train, y_test = train_test_split( 
X, y, test_size=0.2, random_state=42, shuffle=False 
) 

# 4. 训练模型 
print("3. 训练模型...") 
trainer = ModelTrainer() 
models = trainer.train_all_models(X_train, y_train) 

# 5. 评估模型 
print("4. 评估模型...") 
evaluator = ModelEvaluator(models) 
metrics = evaluator.calculate_metrics(X_test, y_test) 
evaluator.plot_predictions(X_test, y_test) 
evaluator.plot_feature_importance(X.columns) 

# 6. 生成预测和调度建议 
print("5. 生成调度建议...") 
optimizer = SchedulingOptimizer() 

# 使用最佳模型进行预测 
best_model_name = max(metrics, key=lambda x: metrics[x]['R2']) 
best_model = models[best_model_name] 

# 预测未来24小时 
future_features = create_future_features() 
predictions = best_model.predict(future_features) 

# 生成调度建议 
recommendations = optimizer.generate_schedule_recommendations( 
predictions, weather_forecast=1 
) 

# 7. 保存结果 
print("6. 保存结果...") 
save_results(models, metrics, predictions, recommendations) 

print("\n=== 项目完成 ===") 
print(f"最佳模型: {best_model_name}") 
print(f"R2分数: {metrics[best_model_name]['R2']:.3f}") 

return models, metrics, recommendations 

if __name__ == "__main__": 
models, metrics, recommendations = main() 
三、关键算法和技术细节 
1. 时间序列特征工程 
python 
# 傅里叶变换提取周期性特征 
def extract_fourier_features(series, period, n_harmonics=3): 
"""提取傅里叶特征""" 
t = np.arange(len(series)) 
features = pd.DataFrame() 

for n in range(1, n_harmonics + 1): 
features[f'fourier_sin_{period}_{n}'] = np.sin(2 * np.pi * n * t / period) 
features[f'fourier_cos_{period}_{n}'] = np.cos(2 * np.pi * n * t / period) 

return features 
2. 模型集成方法 
python 
# 堆叠集成模型 
class StackingEnsemble: 
def __init__(self, base_models, meta_model): 
self.base_models = base_models 
self.meta_model = meta_model 

def fit(self, X, y): 
"""训练堆叠模型""" 
# 第一步：训练基模型 
base_predictions = [] 
for model in self.base_models: 
model.fit(X, y) 
pred = model.predict(X) 
base_predictions.append(pred) 

# 创建元特征 
X_meta = np.column_stack(base_predictions) 

# 第二步：训练元模型 
self.meta_model.fit(X_meta, y) 

def predict(self, X): 
"""使用堆叠模型进行预测""" 
base_preds = [model.predict(X) for model in self.base_models] 
X_meta = np.column_stack(base_preds) 
return self.meta_model.predict(X_meta) 
3. 超参数优化 
python 
# 使用Optuna进行超参数优化 
def optimize_xgboost_params(X_train, y_train, n_trials=100): 
"""使用Optuna优化XGBoost参数""" 

def objective(trial): 
params = { 
'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 
'max_depth': trial.suggest_int('max_depth', 3, 10), 
'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3), 
'subsample': trial.suggest_uniform('subsample', 0.6, 1.0), 
'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0), 
'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0), 
'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0), 
'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0) 
} 

model = xgb.XGBRegressor(**params, random_state=42) 
scores = cross_val_score(model, X_train, y_train, 
cv=5, scoring='r2', n_jobs=-1) 
return scores.mean() 

study = optuna.create_study(direction='maximize') 
study.optimize(objective, n_trials=n_trials) 

return study.best_params 
四、运行说明 
1. 环境配置 
bash 
# 创建虚拟环境 
python -m venv venv 
source venv/bin/activate  # Linux/Mac 
# 或 venv\Scripts\activate  # Windows 

# 安装依赖 
pip install -r requirements.txt 
2. 运行项目 
bash 
# 运行完整流程 
python main.py 

# 运行特定模块 
python -m src.data_preprocessing 
python -m src.model_training 

# 使用Jupyter Notebook进行分析 
jupyter notebook notebooks/外卖订单预测分析.ipynb 
3. 输出文件 
运行后将生成： 
models/ : 训练好的模型文件 
results/predictions.csv : 预测结果 
results/performance_metrics.xlsx : 性能指标 
results/schedule_recommendations.json : 调度建议 
可视化图表文件（PNG格式） 
五、扩展性和维护 
1. 配置文件 ( config.py ) 
python 
# 项目配置 
class Config: 
# 数据配置 
DATA_PATH = 'data/simulated_data.csv' 
SAVE_RESULTS = True 

# 模型配置 
MODELS_TO_TRAIN = ['random_forest', 'xgboost', 'lightgbm'] 
TEST_SIZE = 0.2 
RANDOM_STATE = 42 

# 优化配置 
USE_OPTUNA = False 
N_TRIALS = 100 

# 调度配置 
RIDER_EFFICIENCY = 10  # 单/小时 
MAX_RIDERS_PER_REGION = 50 

# 可视化配置 
PLOT_STYLE = 'seaborn-darkgrid' 
FIGURE_SIZE = (12, 8) 
2. 日志配置 
python 
import logging 

def setup_logging(): 
"""配置日志系统""" 
logging.basicConfig( 
level=logging.INFO, 
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
handlers=[ 
logging.FileHandler('logs/project.log'), 
logging.StreamHandler() 
] 
) 

return logging.getLogger(__name__) 
这个代码框架提供了完整的外卖订单预测和调度优化解决方案，具有模块化、可配置、可扩展的特点，完全符合《Python与大数据分析》课程的要求。
```

---

## 项目简介

本项目是一个完整的**外卖订单量预测与骑手调度优化系统**，使用机器学习算法预测未来订单量，并基于预测结果生成智能调度建议。

### 核心功能

- 📊 **数据生成**：模拟真实外卖订单数据（包含时间、天气、区域等特征）
- 🔍 **探索性数据分析**：全面的EDA分析，包含时间序列、分布、相关性、分类变量和区域分析
- 🔧 **数据预处理**：数据清洗、特征工程、滞后特征、滚动统计
- 🤖 **模型训练**：随机森林、XGBoost、LightGBM多模型训练
- 📈 **模型评估**：MAE、RMSE、R²、MAPE等多维度评估
- 🚴 **调度优化**：基于预测结果的骑手分配与调度建议
- 📓 **交互式分析**：Jupyter Notebook完整分析流程

---

## 项目结构

```
外卖订单预测项目/
├── data/
│   ├── __init__.py
│   ├── simulated_data.csv       # 模拟数据集
│   └── data_generator.py        # 数据生成模块
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # 数据预处理模块
│   ├── eda.py                   # 探索性数据分析模块
│   ├── feature_engineering.py   # 特征工程模块
│   ├── model_training.py        # 模型训练模块
│   ├── model_evaluation.py      # 模型评估模块
│   ├── utils.py                 # 工具
│   └── scheduling_optimization.py # 调度优化模块
├── models/
│   ├── .gitkeep 
│   ├── random_forest_model.pkl  # 随机森林模型(运行后生成)
│   └── xgboost_model.pkl        # XGBoost模型(运行后生成)
├── results/                     # 分析结果
│   ├── .gitkeep 
│   └── eda/                     # EDA分析结果
├── notebooks/
│   └── 外卖订单预测分析.ipynb    # Jupyter Notebook分析文件
├── logs/
│   ├── .gitkeep 
│   └── project.log              # 运行日志(运行后生成)
├── config.py                    # 配置文件
├── main.py                      # 主程序入口
├── requirements.txt             # 依赖包列表
├── .gitignore                   # .gitignore
└── README.md                    # 项目说明文档
```

---

## 快速开始

### 1. 环境配置

**系统要求**：
- Python 3.8+
- pip 或 conda

**安装步骤**：

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 2. 安装依赖包
pip install -r requirements.txt
```

### 2. 运行项目

**完整流程（推荐）**：

```bash
# 运行主程序（包含数据生成、训练、评估、调度优化）
python main.py

# 或使用标准包入口运行
python -m food_delivery_forecast
```

**运行特定模块**：

```bash
# 仅生成数据
python -m data.data_generator

# 仅数据预处理
python -m src.data_preprocessing

# 仅EDA分析
python -m src.eda

# 仅模型训练
python -m src.model_training

# 仅调度优化
python -m src.scheduling_optimization
```

**使用Jupyter Notebook进行交互式分析**：

```bash
# 启动Jupyter Notebook
jupyter notebook

# 在浏览器中打开 notebooks/外卖订单预测分析.ipynb
```

### 3. 查看结果

运行完成后，查看以下文件：

- **模型文件**：`models/` 目录
- **预测结果**：`results/predictions.csv`
- **性能指标**：`results/performance_metrics.xlsx`
- **可视化图表**：`results/*.png`
- **EDA分析结果**：`results/eda/` 目录
- **调度建议**：`results/schedule_recommendations.json`
- **运行日志**：`logs/project.log`

### 4. 配置管理（新增）

项目支持分环境配置，配置文件位于 `config/`：

- `config/settings.base.json`：基础配置
- `config/settings.dev.json`：开发环境覆盖配置
- `config/settings.prod.json`：生产环境覆盖配置

通过环境变量切换：

```bash
export APP_ENV=dev   # 或 prod
python main.py
```

关键项也支持环境变量直接覆盖，例如：

```bash
export TEST_SIZE=0.25
export ENABLE_EDA=false
python main.py
```

### 4. 使用Docker运行

**前提条件**：
- 安装Docker和Docker Compose

**启动步骤**：

```bash
# 在项目根目录执行
docker compose up
```

**查看日志**：

```bash
docker compose logs -f
```

**停止服务**：

```bash
docker compose down
```

**清理容器和镜像**（可选）：

```bash
docker compose down --rmi all
```

---

## 核心算法说明

### 1. 数据生成算法

- **时间模式**：使用正弦函数模拟每日双峰（午餐11-13点，晚餐18-20点）
- **随机性**：泊松分布模拟订单量波动
- **区域分配**：狄利克雷分布模拟各区域订单分布

### 2. 特征工程

| 特征类型 | 具体特征 |
|---------|---------|
| 时间特征 | hour, weekday, is_weekend, is_holiday |
| 滞后特征 | orders_lag_1h, orders_lag_2h, orders_lag_3h, orders_lag_6h, orders_lag_12h |
| 滚动统计 | rolling_mean, rolling_std, rolling_max, rolling_min (窗口: 3h, 6h, 12h, 24h) |
| 周期特征 | hour_sin, hour_cos, weekday_sin, weekday_cos |
| 交互特征 | temp_weather_interaction, hour_weekend_interaction |
| 统计特征 | order_trend, order_acceleration, order_change_rate |

### 3. 模型训练

- **随机森林 (Random Forest)**：集成学习，抗过拟合能力强
- **XGBoost**：梯度提升树，预测精度高
- **LightGBM**（可选）：快速训练，适合大规模数据

### 4. 调度优化

- **K-Means聚类**：识别订单热点区域
- **贪心算法**：优化骑手分配（从富余区域调配到短缺区域）
- **动态调整**：考虑天气、高峰时段等因素

---

## 配置说明

编辑 `config.py` 文件可自定义配置：

```python
class Config:
    # 数据配置
    SIMULATION_DAYS = 90          # 模拟天数
    TEST_SIZE = 0.2               # 测试集比例
    
    # 模型配置
    MODELS_TO_TRAIN = ['random_forest', 'xgboost']  # 训练的模型
    RANDOM_STATE = 42             # 随机种子
    
    # 特征工程配置
    LAG_FEATURES = [1, 2, 3, 6, 12]       # 滞后期
    ROLLING_WINDOWS = [3, 6, 12, 24]      # 滚动窗口
    
    # 调度优化配置
    RIDER_EFFICIENCY = 10         # 骑手效率（单/小时）
    MAX_RIDERS_PER_REGION = 50    # 每区域最大骑手数
```

---

## 性能指标

### 模型评估指标

| 指标 | 说明 | 目标值 |
|-----|------|--------|
| MAE | 平均绝对误差 | < 10单 |
| RMSE | 均方根误差 | < 15单 |
| R² | 决定系数 | > 0.85 |
| MAPE | 平均绝对百分比误差 | < 15% |

### 调度优化指标

- **骑手利用率**：目标 > 80%
- **订单响应时间**：目标 < 30分钟
- **区域覆盖率**：目标 = 100%

---
