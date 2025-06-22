# 导入必要的库
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
np.random.seed(42)

# 定义基础设施类型枚举
class InfrastructureType(Enum):
    VTOL_PLATFORM = "垂直起降平台"
    CONTROL_CENTER = "控制中心"
    GENERAL_AIRPORT = "通用机场"
    CHARGING_STATION = "充电站"
    MAINTENANCE_FACILITY = "维修设施"
    SAFETY_MONITORING = "安全和监控系统"
    TRAINING_CENTER = "培训中心"
    FLIGHT_MANAGEMENT = "飞行管理软件"
    DRONE_FLEET = "无人机群管理平台"
    DATA_ANALYTICS = "数据分析平台"
    CYBERSECURITY = "网络安全系统"
    LICENSING_DATABASE = "许可和认证数据库"
    COMPLIANCE_TOOLS = "合规监控工具"

# 定义收入类型枚举
class RevenueType(Enum):
    LANDING_PARKING = "起降与停场服务收费"
    PASSENGER_SERVICE = "旅客服务费"
    CARGO_HANDLING = "货物处理费"
    FUEL_SERVICE = "航空燃油加注服务"
    CHARGING_FEE = "充电费"
    MAINTENANCE_SERVICE = "维修服务费"
    ATC_SERVICE = "空管服务费"
    COMMERCIAL_INCOME = "商业收入"
    GOVERNMENT_SUBSIDY = "政府补贴"
    SOFTWARE_LICENSE = "软件许可费"
    SYSTEM_ACCESS = "系统授权和访问费"
    DATA_SERVICE = "数据服务费"
    TRAINING_FEE = "培训费"
    CONSULTING_FEE = "咨询服务费"
    CERTIFICATION_FEE = "认证和许可费"
    SUBSCRIPTION_FEE = "订阅费"
    INTEGRATION_SERVICE = "系统集成服务费"

# 收入流数据类
@dataclass
class RevenueStream:
    revenue_type: RevenueType
    base_amount: float  # 年度基础金额（万元）
    growth_rate: float  # 年增长率
    is_stable: bool = True  # 是否稳定收入
    contract_period: int = 12  # 合同期限（月）
    renewal_rate: float = 0.9  # 续约率
    seasonality: Dict[int, float] = field(default_factory=lambda: {i: 1.0 for i in range(1, 13)})  # 季节性因子

# 基础设施收入类
@dataclass
class InfrastructureRevenue:
    asset_type: InfrastructureType
    revenue_streams: List[RevenueStream] = field(default_factory=list)
    
    @classmethod
    def create_default_revenue(cls, infra_type: InfrastructureType) -> 'InfrastructureRevenue':
        infra_revenue = cls(asset_type=infra_type)
        
        if infra_type == InfrastructureType.VTOL_PLATFORM:
            # 垂直起降平台收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.LANDING_PARKING,
                    base_amount=500.0,
                    growth_rate=0.08,
                    seasonality={1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2, 
                                 7: 1.3, 8: 1.3, 9: 1.2, 10: 1.0, 11: 0.9, 12: 0.8}
                ),
                RevenueStream(
                    revenue_type=RevenueType.PASSENGER_SERVICE,
                    base_amount=300.0,
                    growth_rate=0.06,
                    seasonality={1: 0.7, 2: 0.7, 3: 0.8, 4: 0.9, 5: 1.0, 6: 1.2, 
                                 7: 1.4, 8: 1.4, 9: 1.1, 10: 1.0, 11: 0.9, 12: 0.8}
                ),
                RevenueStream(
                    revenue_type=RevenueType.CHARGING_FEE,
                    base_amount=150.0,
                    growth_rate=0.10,
                    is_stable=False,
                    seasonality={1: 0.9, 2: 0.9, 3: 0.9, 4: 1.0, 5: 1.0, 6: 1.1, 
                                 7: 1.2, 8: 1.2, 9: 1.1, 10: 1.0, 11: 0.9, 12: 0.8}
                ),
                RevenueStream(
                    revenue_type=RevenueType.COMMERCIAL_INCOME,
                    base_amount=200.0,
                    growth_rate=0.05,
                    is_stable=False,
                    seasonality={1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.0, 6: 1.1, 
                                 7: 1.3, 8: 1.3, 9: 1.1, 10: 1.0, 11: 0.9, 12: 0.8}
                ),
                RevenueStream(
                    revenue_type=RevenueType.GOVERNMENT_SUBSIDY,
                    base_amount=100.0,
                    growth_rate=0.02,
                    contract_period=36,
                    renewal_rate=0.8,
                    seasonality={i: 1.0 for i in range(1, 13)}
                )
            ]
        elif infra_type == InfrastructureType.CONTROL_CENTER:
            # 控制中心收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.ATC_SERVICE,
                    base_amount=800.0,
                    growth_rate=0.07,
                    contract_period=24,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.SYSTEM_ACCESS,
                    base_amount=400.0,
                    growth_rate=0.09,
                    contract_period=12,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.DATA_SERVICE,
                    base_amount=300.0,
                    growth_rate=0.12,
                    is_stable=False,
                    seasonality={1: 0.9, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 
                                 7: 1.1, 8: 1.1, 9: 1.0, 10: 1.0, 11: 1.0, 12: 0.9}
                ),
                RevenueStream(
                    revenue_type=RevenueType.GOVERNMENT_SUBSIDY,
                    base_amount=200.0,
                    growth_rate=0.03,
                    contract_period=36,
                    renewal_rate=0.85,
                    seasonality={i: 1.0 for i in range(1, 13)}
                )
            ]
        elif infra_type == InfrastructureType.GENERAL_AIRPORT:
            # 通用机场收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.LANDING_PARKING,
                    base_amount=1200.0,
                    growth_rate=0.06,
                    seasonality={1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2, 
                                 7: 1.3, 8: 1.3, 9: 1.2, 10: 1.0, 11: 0.9, 12: 0.8}
                ),
                RevenueStream(
                    revenue_type=RevenueType.PASSENGER_SERVICE,
                    base_amount=800.0,
                    growth_rate=0.07,
                    seasonality={1: 0.7, 2: 0.7, 3: 0.8, 4: 0.9, 5: 1.0, 6: 1.2, 
                                 7: 1.4, 8: 1.4, 9: 1.1, 10: 1.0, 11: 0.9, 12: 0.8}
                ),
                RevenueStream(
                    revenue_type=RevenueType.CARGO_HANDLING,
                    base_amount=500.0,
                    growth_rate=0.08,
                    seasonality={1: 0.9, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 
                                 7: 1.1, 8: 1.1, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.2}
                ),
                RevenueStream(
                    revenue_type=RevenueType.FUEL_SERVICE,
                    base_amount=600.0,
                    growth_rate=0.05,
                    is_stable=False,
                    seasonality={1: 0.9, 2: 0.9, 3: 0.9, 4: 1.0, 5: 1.0, 6: 1.1, 
                                 7: 1.2, 8: 1.2, 9: 1.1, 10: 1.0, 11: 0.9, 12: 0.8}
                ),
                RevenueStream(
                    revenue_type=RevenueType.COMMERCIAL_INCOME,
                    base_amount=700.0,
                    growth_rate=0.09,
                    is_stable=False,
                    seasonality={1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.0, 6: 1.1, 
                                 7: 1.3, 8: 1.3, 9: 1.1, 10: 1.0, 11: 0.9, 12: 0.8}
                ),
                RevenueStream(
                    revenue_type=RevenueType.GOVERNMENT_SUBSIDY,
                    base_amount=300.0,
                    growth_rate=0.02,
                    contract_period=36,
                    renewal_rate=0.8,
                    seasonality={i: 1.0 for i in range(1, 13)}
                )
            ]
        elif infra_type == InfrastructureType.CHARGING_STATION:
            # 充电站收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.CHARGING_FEE,
                    base_amount=400.0,
                    growth_rate=0.12,
                    is_stable=False,
                    seasonality={1: 0.9, 2: 0.9, 3: 0.9, 4: 1.0, 5: 1.0, 6: 1.1, 
                                 7: 1.2, 8: 1.2, 9: 1.1, 10: 1.0, 11: 0.9, 12: 0.8}
                ),
                RevenueStream(
                    revenue_type=RevenueType.COMMERCIAL_INCOME,
                    base_amount=100.0,
                    growth_rate=0.08,
                    is_stable=False,
                    seasonality={1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.0, 6: 1.1, 
                                 7: 1.3, 8: 1.3, 9: 1.1, 10: 1.0, 11: 0.9, 12: 0.8}
                ),
                RevenueStream(
                    revenue_type=RevenueType.GOVERNMENT_SUBSIDY,
                    base_amount=150.0,
                    growth_rate=0.04,
                    contract_period=24,
                    renewal_rate=0.75,
                    seasonality={i: 1.0 for i in range(1, 13)}
                )
            ]
        elif infra_type == InfrastructureType.MAINTENANCE_FACILITY:
            # 维修设施收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.MAINTENANCE_SERVICE,
                    base_amount=600.0,
                    growth_rate=0.07,
                    seasonality={1: 0.9, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 
                                 7: 1.1, 8: 1.1, 9: 1.0, 10: 1.0, 11: 1.0, 12: 0.9}
                ),
                RevenueStream(
                    revenue_type=RevenueType.CONSULTING_FEE,
                    base_amount=200.0,
                    growth_rate=0.09,
                    contract_period=6,
                    renewal_rate=0.8,
                    seasonality={1: 1.0, 2: 1.0, 3: 1.1, 4: 1.1, 5: 1.0, 6: 1.0, 
                                 7: 1.0, 8: 1.0, 9: 1.1, 10: 1.1, 11: 1.0, 12: 0.9}
                ),
                RevenueStream(
                    revenue_type=RevenueType.TRAINING_FEE,
                    base_amount=150.0,
                    growth_rate=0.08,
                    contract_period=3,
                    renewal_rate=0.85,
                    seasonality={1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9, 6: 0.8, 
                                 7: 0.8, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.2, 12: 1.1}
                )
            ]
        elif infra_type == InfrastructureType.SAFETY_MONITORING:
            # 安全和监控系统收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.SYSTEM_ACCESS,
                    base_amount=500.0,
                    growth_rate=0.08,
                    contract_period=24,
                    renewal_rate=0.9,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.DATA_SERVICE,
                    base_amount=300.0,
                    growth_rate=0.11,
                    contract_period=12,
                    renewal_rate=0.85,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.MAINTENANCE_SERVICE,
                    base_amount=200.0,
                    growth_rate=0.06,
                    contract_period=12,
                    renewal_rate=0.95,
                    seasonality={i: 1.0 for i in range(1, 13)}
                )
            ]
        elif infra_type == InfrastructureType.TRAINING_CENTER:
            # 培训中心收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.TRAINING_FEE,
                    base_amount=400.0,
                    growth_rate=0.09,
                    is_stable=False,
                    seasonality={1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9, 6: 0.8, 
                                 7: 0.8, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.2, 12: 1.1}
                ),
                RevenueStream(
                    revenue_type=RevenueType.CERTIFICATION_FEE,
                    base_amount=250.0,
                    growth_rate=0.07,
                    contract_period=6,
                    renewal_rate=0.8,
                    seasonality={1: 1.1, 2: 1.1, 3: 1.0, 4: 1.0, 5: 0.9, 6: 0.9, 
                                 7: 0.9, 8: 0.9, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.1}
                ),
                RevenueStream(
                    revenue_type=RevenueType.CONSULTING_FEE,
                    base_amount=150.0,
                    growth_rate=0.08,
                    is_stable=False,
                    seasonality={1: 1.0, 2: 1.0, 3: 1.1, 4: 1.1, 5: 1.0, 6: 1.0, 
                                 7: 1.0, 8: 1.0, 9: 1.1, 10: 1.1, 11: 1.0, 12: 0.9}
                )
            ]
        elif infra_type == InfrastructureType.FLIGHT_MANAGEMENT:
            # 飞行管理软件收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.SOFTWARE_LICENSE,
                    base_amount=700.0,
                    growth_rate=0.10,
                    contract_period=12,
                    renewal_rate=0.9,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.SUBSCRIPTION_FEE,
                    base_amount=500.0,
                    growth_rate=0.12,
                    contract_period=12,
                    renewal_rate=0.85,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.INTEGRATION_SERVICE,
                    base_amount=300.0,
                    growth_rate=0.08,
                    is_stable=False,
                    seasonality={1: 1.1, 2: 1.1, 3: 1.0, 4: 1.0, 5: 0.9, 6: 0.9, 
                                 7: 0.9, 8: 0.9, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.1}
                )
            ]
        elif infra_type == InfrastructureType.DRONE_FLEET:
            # 无人机群管理平台收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.SYSTEM_ACCESS,
                    base_amount=600.0,
                    growth_rate=0.11,
                    contract_period=12,
                    renewal_rate=0.85,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.DATA_SERVICE,
                    base_amount=400.0,
                    growth_rate=0.13,
                    contract_period=6,
                    renewal_rate=0.8,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.MAINTENANCE_SERVICE,
                    base_amount=250.0,
                    growth_rate=0.07,
                    is_stable=False,
                    seasonality={1: 0.9, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 
                                 7: 1.1, 8: 1.1, 9: 1.0, 10: 1.0, 11: 1.0, 12: 0.9}
                )
            ]
        elif infra_type == InfrastructureType.DATA_ANALYTICS:
            # 数据分析平台收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.DATA_SERVICE,
                    base_amount=800.0,
                    growth_rate=0.14,
                    contract_period=12,
                    renewal_rate=0.85,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.SUBSCRIPTION_FEE,
                    base_amount=600.0,
                    growth_rate=0.12,
                    contract_period=12,
                    renewal_rate=0.9,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.CONSULTING_FEE,
                    base_amount=300.0,
                    growth_rate=0.09,
                    is_stable=False,
                    seasonality={1: 1.0, 2: 1.0, 3: 1.1, 4: 1.1, 5: 1.0, 6: 1.0, 
                                 7: 1.0, 8: 1.0, 9: 1.1, 10: 1.1, 11: 1.0, 12: 0.9}
                )
            ]
        elif infra_type == InfrastructureType.CYBERSECURITY:
            # 网络安全系统收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.SYSTEM_ACCESS,
                    base_amount=700.0,
                    growth_rate=0.09,
                    contract_period=24,
                    renewal_rate=0.95,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.SUBSCRIPTION_FEE,
                    base_amount=500.0,
                    growth_rate=0.11,
                    contract_period=12,
                    renewal_rate=0.9,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.CONSULTING_FEE,
                    base_amount=250.0,
                    growth_rate=0.08,
                    is_stable=False,
                    seasonality={1: 1.0, 2: 1.0, 3: 1.1, 4: 1.1, 5: 1.0, 6: 1.0, 
                                 7: 1.0, 8: 1.0, 9: 1.1, 10: 1.1, 11: 1.0, 12: 0.9}
                )
            ]
        elif infra_type == InfrastructureType.LICENSING_DATABASE:
            # 许可和认证数据库收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.CERTIFICATION_FEE,
                    base_amount=500.0,
                    growth_rate=0.07,
                    contract_period=36,
                    renewal_rate=0.9,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.DATA_SERVICE,
                    base_amount=400.0,
                    growth_rate=0.10,
                    contract_period=12,
                    renewal_rate=0.85,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.SYSTEM_ACCESS,
                    base_amount=300.0,
                    growth_rate=0.08,
                    contract_period=12,
                    renewal_rate=0.9,
                    seasonality={i: 1.0 for i in range(1, 13)}
                )
            ]
        elif infra_type == InfrastructureType.COMPLIANCE_TOOLS:
            # 合规监控工具收入
            infra_revenue.revenue_streams = [
                RevenueStream(
                    revenue_type=RevenueType.SOFTWARE_LICENSE,
                    base_amount=600.0,
                    growth_rate=0.09,
                    contract_period=12,
                    renewal_rate=0.9,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.SUBSCRIPTION_FEE,
                    base_amount=400.0,
                    growth_rate=0.11,
                    contract_period=12,
                    renewal_rate=0.85,
                    seasonality={i: 1.0 for i in range(1, 13)}
                ),
                RevenueStream(
                    revenue_type=RevenueType.CONSULTING_FEE,
                    base_amount=200.0,
                    growth_rate=0.08,
                    is_stable=False,
                    seasonality={1: 1.0, 2: 1.0, 3: 1.1, 4: 1.1, 5: 1.0, 6: 1.0, 
                                 7: 1.0, 8: 1.0, 9: 1.1, 10: 1.1, 11: 1.0, 12: 0.9}
                )
            ]
        
        # 添加收入稳定性评估方法
        def get_revenue_stability_metrics(self) -> Dict[str, float]:
            # 计算平均稳定性
            stability_scores = [1.0 if stream.is_stable else 0.6 for stream in self.revenue_streams]
            weights = [stream.base_amount for stream in self.revenue_streams]
            total_weight = sum(weights)
            weighted_stability = sum(score * weight for score, weight in zip(stability_scores, weights)) / total_weight if total_weight > 0 else 0
            
            # 计算稳定性标准差
            stability_std = np.std(stability_scores) if len(stability_scores) > 1 else 0
            
            # 计算收入增长率（加权平均）
            growth_rates = [stream.growth_rate for stream in self.revenue_streams]
            weighted_growth = sum(rate * weight for rate, weight in zip(growth_rates, weights)) / total_weight if total_weight > 0 else 0
            
            # 计算波动性（基于季节性因子）
            volatility = 0
            for stream in self.revenue_streams:
                seasonal_values = list(stream.seasonality.values())
                stream_volatility = np.std(seasonal_values) / np.mean(seasonal_values) if np.mean(seasonal_values) > 0 else 0
                volatility += stream_volatility * (stream.base_amount / total_weight) if total_weight > 0 else 0
            
            return {
                "average_stability": weighted_stability,
                "stability_std": stability_std,
                "revenue_growth": weighted_growth,
                "volatility": volatility
            }
        
        # 将方法添加到实例
        infra_revenue.get_revenue_stability_metrics = get_revenue_stability_metrics.__get__(infra_revenue)
        
        return infra_revenue

# 低空经济基础设施资产类（占位符）
class LowAltitudeInfrastructureAsset:
    def __init__(self, location_tier=1):
        self.location_tier = location_tier  # 1-一线城市, 2-二线城市, 3-三线城市及以下

# 风险评估系统
class RiskAssessmentSystem:
    def __init__(self, asset, revenue):
        self.asset = asset
        self.revenue = revenue
    
    def calculate_market_risk(self) -> float:
        # 市场风险评估
        base_risk = 0.4  # 基础市场风险
        
        # 根据资产类型调整风险
        asset_type_risk_map = {
            InfrastructureType.VTOL_PLATFORM: 0.5,
            InfrastructureType.CONTROL_CENTER: 0.4,
            InfrastructureType.GENERAL_AIRPORT: 0.6,
            InfrastructureType.CHARGING_STATION: 0.3,
            InfrastructureType.MAINTENANCE_FACILITY: 0.4,
            InfrastructureType.SAFETY_MONITORING: 0.2,
            InfrastructureType.TRAINING_CENTER: 0.3,
            InfrastructureType.FLIGHT_MANAGEMENT: 0.2,
            InfrastructureType.DRONE_FLEET: 0.4,
            InfrastructureType.DATA_ANALYTICS: 0.3,
            InfrastructureType.CYBERSECURITY: 0.2,
            InfrastructureType.LICENSING_DATABASE: 0.1,
            InfrastructureType.COMPLIANCE_TOOLS: 0.2
        }
        asset_type_risk = asset_type_risk_map.get(self.revenue.asset_type, 0.5)
        
        # 根据地理位置调整风险
        location_risk = 0.2 if self.asset.location_tier == 1 else 0.4 if self.asset.location_tier == 2 else 0.6
        
        # 根据收入稳定性调整风险
        stability_metrics = self.revenue.get_revenue_stability_metrics()
        stability_risk = 1 - stability_metrics["average_stability"]
        
        # 综合计算市场风险
        market_risk = base_risk * 0.3 + asset_type_risk * 0.3 + location_risk * 0.2 + stability_risk * 0.2
        return min(max(market_risk, 0.1), 0.9)  # 限制在0.1-0.9范围内
    
    def calculate_credit_risk(self) -> float:
        # 信用风险评估
        base_risk = 0.3  # 基础信用风险
        
        # 根据收入类型调整风险
        govt_subsidy_ratio = sum(stream.base_amount for stream in self.revenue.revenue_streams 
                               if stream.revenue_type == RevenueType.GOVERNMENT_SUBSIDY) / \
                           sum(stream.base_amount for stream in self.revenue.revenue_streams) \
                           if self.revenue.revenue_streams else 0
        
        credit_risk = base_risk * (1 - govt_subsidy_ratio * 0.5)  # 政府补贴比例越高，信用风险越低
        
        # 根据合同期限调整风险
        avg_contract_period = np.mean([stream.contract_period for stream in self.revenue.revenue_streams 
                                     if stream.is_stable]) if self.revenue.revenue_streams else 12
        contract_risk_factor = 1 - min(avg_contract_period / 36, 1)  # 合同期越长，风险越低
        
        credit_risk = credit_risk * 0.7 + contract_risk_factor * 0.3
        return min(max(credit_risk, 0.1), 0.9)  # 限制在0.1-0.9范围内
    
    def calculate_operational_risk(self) -> float:
        # 运营风险评估
        base_risk = 0.5  # 基础运营风险
        
        # 根据资产类型调整风险
        physical_asset_types = [InfrastructureType.VTOL_PLATFORM, InfrastructureType.GENERAL_AIRPORT, 
                               InfrastructureType.CHARGING_STATION, InfrastructureType.MAINTENANCE_FACILITY]
        software_asset_types = [InfrastructureType.FLIGHT_MANAGEMENT, InfrastructureType.DATA_ANALYTICS, 
                              InfrastructureType.CYBERSECURITY, InfrastructureType.LICENSING_DATABASE, 
                              InfrastructureType.COMPLIANCE_TOOLS]
        
        if self.revenue.asset_type in physical_asset_types:
            asset_type_risk = 0.6  # 物理资产运营风险较高
        elif self.revenue.asset_type in software_asset_types:
            asset_type_risk = 0.3  # 软件资产运营风险较低
        else:
            asset_type_risk = 0.4  # 其他类型
        
        # 根据收入多样性调整风险
        revenue_diversity = min(len(self.revenue.revenue_streams) / 5, 1)  # 收入流越多，风险越低
        diversity_risk_factor = 1 - revenue_diversity
        
        operational_risk = base_risk * 0.4 + asset_type_risk * 0.4 + diversity_risk_factor * 0.2
        return min(max(operational_risk, 0.1), 0.9)  # 限制在0.1-0.9范围内
    
    def calculate_policy_risk(self) -> float:
        # 政策风险评估
        base_risk = 0.4  # 基础政策风险
        
        # 根据资产类型调整风险
        regulated_asset_types = [InfrastructureType.VTOL_PLATFORM, InfrastructureType.CONTROL_CENTER, 
                                InfrastructureType.GENERAL_AIRPORT]
        compliance_asset_types = [InfrastructureType.LICENSING_DATABASE, InfrastructureType.COMPLIANCE_TOOLS]
        
        if self.revenue.asset_type in regulated_asset_types:
            asset_type_risk = 0.7  # 高度监管的资产政策风险高
        elif self.revenue.asset_type in compliance_asset_types:
            asset_type_risk = 0.2  # 合规相关资产政策风险低
        else:
            asset_type_risk = 0.4  # 其他类型
        
        # 根据政府补贴比例调整风险
        govt_subsidy_ratio = sum(stream.base_amount for stream in self.revenue.revenue_streams 
                               if stream.revenue_type == RevenueType.GOVERNMENT_SUBSIDY) / \
                           sum(stream.base_amount for stream in self.revenue.revenue_streams) \
                           if self.revenue.revenue_streams else 0
        
        subsidy_risk_factor = govt_subsidy_ratio * 0.6  # 补贴比例越高，政策变动风险越高
        
        policy_risk = base_risk * 0.3 + asset_type_risk * 0.5 + subsidy_risk_factor * 0.2
        return min(max(policy_risk, 0.1), 0.9)  # 限制在0.1-0.9范围内
    
    def calculate_liquidity_risk(self) -> float:
        # 流动性风险评估
        base_risk = 0.3  # 基础流动性风险
        
        # 根据收入稳定性调整风险
        stability_metrics = self.revenue.get_revenue_stability_metrics()
        stability_risk = 1 - stability_metrics["average_stability"]
        
        # 根据收入增长率调整风险
        growth_risk_factor = 1 - min(stability_metrics["revenue_growth"] / 0.15, 1)  # 增长率越高，流动性风险越低
        
        # 根据波动性调整风险
        volatility_risk_factor = min(stability_metrics["volatility"] * 5, 1)  # 波动性越高，流动性风险越高
        
        liquidity_risk = base_risk * 0.3 + stability_risk * 0.3 + growth_risk_factor * 0.2 + volatility_risk_factor * 0.2
        return min(max(liquidity_risk, 0.1), 0.9)  # 限制在0.1-0.9范围内
    
    def get_overall_risk_assessment(self) -> Dict[str, Union[float, str, Dict[str, float]]]:
        # 计算各类风险
        market_risk = self.calculate_market_risk()
        credit_risk = self.calculate_credit_risk()
        operational_risk = self.calculate_operational_risk()
        policy_risk = self.calculate_policy_risk()
        liquidity_risk = self.calculate_liquidity_risk()
        
        # 计算综合风险评分（加权平均）
        weights = {"市场风险": 0.25, "信用风险": 0.2, "运营风险": 0.2, "政策风险": 0.15, "流动性风险": 0.2}
        detailed_scores = {"市场风险": market_risk, "信用风险": credit_risk, "运营风险": operational_risk, 
                         "政策风险": policy_risk, "流动性风险": liquidity_risk}
        
        overall_score = sum(score * weights[risk_type] for risk_type, score in detailed_scores.items())
        
        # 确定风险等级
        if overall_score < 0.4:
            risk_level = "低风险"
        elif overall_score < 0.7:
            risk_level = "中等风险"
        else:
            risk_level = "高风险"
        
        return {
            "overall_risk_score": overall_score,
            "risk_level": risk_level,
            "detailed_scores": detailed_scores
        }

# 现金流模拟函数（增强波动性）
def simulate_cash_flows(infra_revenue: 'InfrastructureRevenue', simulations: int = 1000) -> np.ndarray:
    """使用蒙特卡洛方法模拟基础设施的现金流"""
    # 常量定义
    MONTHS_PER_YEAR = 12
    FORECAST_YEARS = 10
    FORECAST_MONTHS = FORECAST_YEARS * MONTHS_PER_YEAR
    
    cash_flows = np.zeros((FORECAST_MONTHS, simulations))
    
    for stream in infra_revenue.revenue_streams:
        # 初始金额（月度）
        monthly_base = stream.base_amount / MONTHS_PER_YEAR
        stream_cash_flows = np.zeros((FORECAST_MONTHS, simulations))
        
        if stream.is_stable:
            # 稳定收入：考虑合同期限和续约率
            contract_months = stream.contract_period
            renewal_prob = stream.renewal_rate
            current_contract_end = contract_months
            
            for t in range(FORECAST_MONTHS):
                if t >= current_contract_end:
                    renew = np.random.binomial(1, renewal_prob, simulations)
                    if np.mean(renew) > 0:
                        current_contract_end += contract_months
                    else:
                        continue
                
                year = t // MONTHS_PER_YEAR
                month = (t % MONTHS_PER_YEAR) + 1
                growth_factor = (1 + stream.growth_rate) ** year
                seasonal_factor = stream.seasonality.get(month, 1.0)
                growth_noise = np.random.normal(0, 0.02, simulations)  # 2%随机波动
                monthly_amount = monthly_base * growth_factor * (1 + growth_noise) * seasonal_factor
                stream_cash_flows[t] = monthly_amount * (t < current_contract_end)
        else:
            # 不稳定收入：直接按增长率和季节性调整
            for t in range(FORECAST_MONTHS):
                year = t // MONTHS_PER_YEAR
                month = (t % MONTHS_PER_YEAR) + 1
                growth_factor = (1 + stream.growth_rate) ** year
                seasonal_factor = stream.seasonality.get(month, 1.0)
                growth_noise = np.random.normal(0, 0.02, simulations)
                monthly_amount = monthly_base * growth_factor * (1 + growth_noise) * seasonal_factor
                stream_cash_flows[t] = monthly_amount
        
        cash_flows += stream_cash_flows
    
    return cash_flows

# 压力测试类
class CashflowStressTest:
    def __init__(self, infra_revenue, risk_system):
        self.infra_revenue = infra_revenue
        self.risk_system = risk_system
        self.MONTHS_PER_YEAR = 12
        self.FORECAST_YEARS = 10
        self.FORECAST_MONTHS = self.FORECAST_YEARS * self.MONTHS_PER_YEAR
    
    def run_stress_test(self) -> Dict[str, float]:
        # 基准情景
        base_cash_flows = simulate_cash_flows(self.infra_revenue, simulations=1000)
        base_npv = self.calculate_npv(base_cash_flows)
        
        # 轻度衰退情景
        mild_recession_cash_flows = self.apply_stress_scenario(base_cash_flows, severity=0.2)
        mild_npv = self.calculate_npv(mild_recession_cash_flows)
        
        # 中度衰退情景
        moderate_recession_cash_flows = self.apply_stress_scenario(base_cash_flows, severity=0.4)
        moderate_npv = self.calculate_npv(moderate_recession_cash_flows)
        
        # 严重衰退情景
        severe_recession_cash_flows = self.apply_stress_scenario(base_cash_flows, severity=0.7)
        severe_npv = self.calculate_npv(severe_recession_cash_flows)
        
        # 计算预期损失
        expected_loss = base_npv.mean() - (mild_npv.mean() * 0.6 + moderate_npv.mean() * 0.3 + severe_npv.mean() * 0.1)
        
        # 计算VaR
        var_95 = base_npv.mean() - np.percentile(base_npv, 5)
        var_99 = base_npv.mean() - np.percentile(base_npv, 1)
        
        # 计算最大回撤
        max_drawdown = self.calculate_max_drawdown(base_cash_flows)
        
        # 计算恢复概率
        recovery_probability = self.calculate_recovery_probability(moderate_recession_cash_flows)
        
        return {
            "expected_loss": expected_loss,
            "var_95": var_95,
            "var_99": var_99,
            "max_drawdown": max_drawdown,
            "recovery_probability": recovery_probability
        }
    
    def apply_stress_scenario(self, base_cash_flows, severity):
        # 应用压力情景
        stress_cash_flows = base_cash_flows.copy()
        
        # 根据风险评估调整压力程度
        risk_assessment = self.risk_system.get_overall_risk_assessment()
        risk_multiplier = risk_assessment["overall_risk_score"] * 1.5  # 风险越高，压力影响越大
        
        # 应用压力因子
        for t in range(self.FORECAST_MONTHS):
            year = t // self.MONTHS_PER_YEAR
            # 前3年压力最大，之后逐渐恢复
            if year < 3:
                stress_factor = severity * risk_multiplier
            else:
                stress_factor = severity * risk_multiplier * max(0, 1 - (year - 3) * 0.2)
            
            # 应用随机波动
            stress_noise = np.random.normal(0, 0.05, stress_cash_flows.shape[1])  # 5%随机波动
            stress_cash_flows[t] = stress_cash_flows[t] * (1 - stress_factor - stress_noise * stress_factor)
        
        return stress_cash_flows
    
    def calculate_npv(self, cash_flows, discount_rate=0.05):
        # 计算NPV
        npv = np.zeros(cash_flows.shape[1])
        for t in range(self.FORECAST_MONTHS):
            discount_factor = (1 + discount_rate) ** (t / self.MONTHS_PER_YEAR)
            npv += cash_flows[t] / discount_factor
        return npv
    
    def calculate_max_drawdown(self, cash_flows):
        # 计算最大回撤
        cumulative_flows = np.cumsum(cash_flows, axis=0)
        max_drawdowns = np.zeros(cash_flows.shape[1])
        
        for i in range(cash_flows.shape[1]):
            peak = cumulative_flows[0, i]
            max_drawdown = 0
            
            for t in range(1, self.FORECAST_MONTHS):
                if cumulative_flows[t, i] > peak:
                    peak = cumulative_flows[t, i]
                else:
                    drawdown = peak - cumulative_flows[t, i]
                    max_drawdown = max(max_drawdown, drawdown)
            
            max_drawdowns[i] = max_drawdown
        
        return np.mean(max_drawdowns)
    
    def calculate_recovery_probability(self, stress_cash_flows):
        # 计算恢复概率（从压力情景中恢复到基准水平的概率）
        cumulative_flows = np.cumsum(stress_cash_flows, axis=0)
        recovery_count = 0
        
        for i in range(stress_cash_flows.shape[1]):
            # 检查后半期是否恢复到前期水平
            first_half_avg = np.mean(stress_cash_flows[:self.FORECAST_MONTHS//2, i])
            second_half_avg = np.mean(stress_cash_flows[self.FORECAST_MONTHS//2:, i])
            
            if second_half_avg >= first_half_avg * 0.9:  # 恢复到90%以上视为恢复
                recovery_count += 1
        
        return recovery_count / stress_cash_flows.shape[1]

# 数据文件路径
file_paths = {
    "CPI": r"D:\桌面\data\economic_indicators.xlsx",
    "GDP": r"D:\桌面\data\economic_indicators.xlsx",
    "MONEY": r"D:\桌面\data\economic_indicators.xlsx",
    "UNEMPLOYMENT": r"D:\桌面\data\economic_indicators.xlsx",
    "HOUSING": r"D:\桌面\data\economic_indicators.xlsx",
    "SHIBOR": r"D:\桌面\data\interest_rates.xlsx",
    "YIELD": r"D:\桌面\data\interest_rates.xlsx"
}

# 预测时间常量
MONTHS_PER_YEAR = 12
DAYS_PER_YEAR = 252
FORECAST_YEARS = 10
FORECAST_MONTHS = FORECAST_YEARS * MONTHS_PER_YEAR
FORECAST_DAYS = FORECAST_YEARS * DAYS_PER_YEAR

# 图像输出目录
OUTPUT_DIR = r"D:\桌面\111"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)