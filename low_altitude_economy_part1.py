# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from enum import Enum
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.stats import norm
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出目录
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义基础设施类型枚举
class InfrastructureType(Enum):
    AIRPORT = "机场"
    VERTIPORT = "垂直起降场"
    CHARGING = "充电设施"
    MAINTENANCE = "维修设施"
    CONTROL = "空中交通管制"

# 定义收入类型枚举
class RevenueType(Enum):
    LANDING_FEE = "起降费"
    PASSENGER_FEE = "旅客服务费"
    CARGO_FEE = "货物处理费"
    PARKING_FEE = "停机费"
    CHARGING_FEE = "充电费"
    MAINTENANCE_FEE = "维修服务费"
    CONTROL_FEE = "空管服务费"
    COMMERCIAL = "商业收入"
    SUBSIDY = "政府补贴"

# 风险评估函数
def assess_risk(infra_type, revenue_stability, economic_indicators, cash_flows):
    # 初始化风险评分
    risk_scores = {
        "市场风险": 0.0,
        "政策风险": 0.0,
        "运营风险": 0.0,
        "财务风险": 0.0,
        "技术风险": 0.0
    }
    
    # 基于基础设施类型的基础风险评分
    base_risks = {
        InfrastructureType.AIRPORT: {"市场风险": 0.6, "政策风险": 0.4, "运营风险": 0.5, "财务风险": 0.7, "技术风险": 0.3},
        InfrastructureType.VERTIPORT: {"市场风险": 0.7, "政策风险": 0.6, "运营风险": 0.6, "财务风险": 0.8, "技术风险": 0.7},
        InfrastructureType.CHARGING: {"市场风险": 0.5, "政策风险": 0.3, "运营风险": 0.4, "财务风险": 0.5, "技术风险": 0.6},
        InfrastructureType.MAINTENANCE: {"市场风险": 0.4, "政策风险": 0.3, "运营风险": 0.7, "财务风险": 0.6, "技术风险": 0.5},
        InfrastructureType.CONTROL: {"市场风险": 0.3, "政策风险": 0.7, "运营风险": 0.8, "财务风险": 0.5, "技术风险": 0.8}
    }
    
    # 复制基础风险评分
    for risk_type, score in base_risks[infra_type].items():
        risk_scores[risk_type] = score
    
    # 基于收入稳定性调整市场风险和财务风险
    stability_factor = 1 - revenue_stability['average_stability']
    risk_scores["市场风险"] *= (0.7 + 0.6 * stability_factor)
    risk_scores["财务风险"] *= (0.8 + 0.4 * stability_factor)
    
    # 基于经济指标调整政策风险和市场风险
    if economic_indicators is not None and 'GDP' in economic_indicators.columns:
        gdp_growth = economic_indicators['GDP'].pct_change().mean()
        unemployment_rate = economic_indicators['预测失业率'].mean() if '预测失业率' in economic_indicators.columns else 0.05
        
        # GDP增长率高，降低政策风险；失业率高，增加政策风险
        risk_scores["政策风险"] *= (1 - gdp_growth * 2 + unemployment_rate * 3)
        risk_scores["市场风险"] *= (1 - gdp_growth * 3 + unemployment_rate * 2)
    
    # 基于现金流调整运营风险和财务风险
    if cash_flows is not None and len(cash_flows) > 0:
        cf_volatility = np.std(cash_flows) / np.mean(cash_flows) if np.mean(cash_flows) > 0 else 1.0
        cf_trend = np.polyfit(range(len(cash_flows)), cash_flows, 1)[0] / np.mean(cash_flows) if np.mean(cash_flows) > 0 else 0.0
        
        # 现金流波动大，增加运营风险和财务风险；现金流趋势向上，降低财务风险
        risk_scores["运营风险"] *= (1 + cf_volatility * 0.5)
        risk_scores["财务风险"] *= (1 + cf_volatility * 0.7 - cf_trend * 0.5)
    
    # 确保风险评分在0-1之间
    for risk_type in risk_scores:
        risk_scores[risk_type] = max(0, min(1, risk_scores[risk_type]))
    
    # 计算综合风险评分（加权平均）
    weights = {"市场风险": 0.25, "政策风险": 0.2, "运营风险": 0.2, "财务风险": 0.25, "技术风险": 0.1}
    overall_risk_score = sum(risk_scores[risk_type] * weight for risk_type, weight in weights.items())
    
    return {
        "overall_risk_score": overall_risk_score,
        "detailed_scores": risk_scores
    }

# 现金流模拟函数
def simulate_cash_flows(infra_type, base_revenue, years, interest_rate_paths, economic_forecasts=None, risk_level=0.5):
    # 设置月度时间点
    months = years * 12
    time_points = np.arange(months)
    
    # 基于基础设施类型设置收入增长率和波动率
    growth_rates = {
        InfrastructureType.AIRPORT: 0.05,
        InfrastructureType.VERTIPORT: 0.08,
        InfrastructureType.CHARGING: 0.06,
        InfrastructureType.MAINTENANCE: 0.04,
        InfrastructureType.CONTROL: 0.03
    }
    
    volatilities = {
        InfrastructureType.AIRPORT: 0.15,
        InfrastructureType.VERTIPORT: 0.25,
        InfrastructureType.CHARGING: 0.18,
        InfrastructureType.MAINTENANCE: 0.12,
        InfrastructureType.CONTROL: 0.10
    }
    
    # 基于基础设施类型设置成本结构
    cost_structures = {
        InfrastructureType.AIRPORT: {"fixed": 0.4, "variable": 0.3, "maintenance": 0.15, "staff": 0.15},
        InfrastructureType.VERTIPORT: {"fixed": 0.3, "variable": 0.25, "maintenance": 0.2, "staff": 0.25},
        InfrastructureType.CHARGING: {"fixed": 0.5, "variable": 0.3, "maintenance": 0.1, "staff": 0.1},
        InfrastructureType.MAINTENANCE: {"fixed": 0.35, "variable": 0.2, "maintenance": 0.15, "staff": 0.3},
        InfrastructureType.CONTROL: {"fixed": 0.45, "variable": 0.15, "maintenance": 0.1, "staff": 0.3}
    }
    
    # 获取当前基础设施类型的参数
    annual_growth = growth_rates[infra_type]
    monthly_growth = (1 + annual_growth) ** (1/12) - 1
    annual_vol = volatilities[infra_type]
    monthly_vol = annual_vol / np.sqrt(12)
    cost_structure = cost_structures[infra_type]
    
    # 初始化现金流数组
    revenues = np.zeros(months)
    costs = np.zeros(months)
    cash_flows = np.zeros(months)
    
    # 设置初始收入和成本
    initial_revenue = base_revenue
    initial_cost = base_revenue * 0.7  # 假设初始成本是收入的70%
    
    # 如果有经济预测数据，调整增长率和波动率
    if economic_forecasts is not None:
        if 'GDP' in economic_forecasts.columns:
            # 使用GDP增长率调整收入增长率
            gdp_growth = economic_forecasts['GDP'].pct_change().fillna(0).values
            gdp_growth = np.append(gdp_growth, [gdp_growth[-1]] * (months - len(gdp_growth))) if len(gdp_growth) < months else gdp_growth[:months]
            monthly_growth_adjustments = gdp_growth * 0.5  # GDP增长率对收入增长的影响因子
        
        if '预测失业率' in economic_forecasts.columns:
            # 使用失业率调整收入波动率
            unemployment = economic_forecasts['预测失业率'].values
            unemployment = np.append(unemployment, [unemployment[-1]] * (months - len(unemployment))) if len(unemployment) < months else unemployment[:months]
            monthly_vol_adjustments = unemployment * 0.2  # 失业率对收入波动的影响因子
    else:
        monthly_growth_adjustments = np.zeros(months)
        monthly_vol_adjustments = np.zeros(months)
    
    # 模拟收入路径
    revenues[0] = initial_revenue
    for t in range(1, months):
        # 调整后的月度增长率和波动率
        adjusted_growth = monthly_growth + monthly_growth_adjustments[t-1] if t-1 < len(monthly_growth_adjustments) else monthly_growth
        adjusted_vol = monthly_vol + monthly_vol_adjustments[t-1] if t-1 < len(monthly_vol_adjustments) else monthly_vol
        
        # 生成随机收入增长
        shock = np.random.normal(0, adjusted_vol)
        revenues[t] = revenues[t-1] * (1 + adjusted_growth + shock)
    
    # 模拟成本路径
    costs[0] = initial_cost
    for t in range(1, months):
        # 固定成本增长较慢，变动成本与收入相关
        fixed_cost = costs[0] * cost_structure["fixed"] * (1 + monthly_growth * 0.5) ** t
        variable_cost = revenues[t] * cost_structure["variable"]
        maintenance_cost = costs[0] * cost_structure["maintenance"] * (1 + monthly_growth * 0.7) ** t
        staff_cost = costs[0] * cost_structure["staff"] * (1 + monthly_growth * 0.8) ** t
        
        # 利率对成本的影响（主要是固定成本中的融资成本）
        if interest_rate_paths is not None and t < len(interest_rate_paths):
            interest_effect = interest_rate_paths[t] * 0.1 * fixed_cost
        else:
            interest_effect = 0
        
        costs[t] = fixed_cost + variable_cost + maintenance_cost + staff_cost + interest_effect
    
    # 计算现金流
    cash_flows = revenues - costs
    
    # 添加风险调整
    if risk_level > 0:
        # 风险冲击：随机选择几个月份，显著降低收入
        num_shocks = int(months * risk_level * 0.2)  # 风险级别越高，冲击越多
        shock_months = np.random.choice(range(months), size=num_shocks, replace=False)
        shock_severity = np.random.uniform(0.1, 0.5, size=num_shocks)  # 10%-50%的收入下降
        
        for i, month in enumerate(shock_months):
            revenues[month] *= (1 - shock_severity[i])
            cash_flows[month] = revenues[month] - costs[month]
    
    return revenues, costs, cash_flows