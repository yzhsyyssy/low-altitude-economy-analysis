# 压力测试函数
def stress_test(infra_type, cash_flows, interest_rate_paths, economic_scenarios=None):
    # 复制原始现金流
    base_cash_flows = cash_flows.copy()
    months = len(base_cash_flows)
    
    # 定义压力情景
    scenarios = {
        "基准情景": {"revenue_shock": 0.0, "cost_increase": 0.0, "interest_shock": 0.0},
        "轻度衰退": {"revenue_shock": -0.15, "cost_increase": 0.05, "interest_shock": 0.01},
        "中度衰退": {"revenue_shock": -0.30, "cost_increase": 0.10, "interest_shock": 0.02},
        "严重衰退": {"revenue_shock": -0.50, "cost_increase": 0.20, "interest_shock": 0.04}
    }
    
    # 基于基础设施类型调整压力情景的影响
    sensitivity = {
        InfrastructureType.AIRPORT: 1.2,  # 机场对经济衰退更敏感
        InfrastructureType.VERTIPORT: 1.3,  # 垂直起降场对经济衰退最敏感
        InfrastructureType.CHARGING: 0.9,  # 充电设施相对较少敏感
        InfrastructureType.MAINTENANCE: 0.8,  # 维修设施相对较少敏感
        InfrastructureType.CONTROL: 0.7   # 空管系统最不敏感
    }
    
    # 应用基础设施类型的敏感度调整
    for scenario in scenarios:
        if scenario != "基准情景":
            scenarios[scenario]["revenue_shock"] *= sensitivity[infra_type]
    
    # 初始化结果存储
    stressed_cash_flows = {}
    npv_results = {}
    discount_rate = 0.08  # 基准折现率
    
    # 对每个情景进行压力测试
    for scenario, shocks in scenarios.items():
        # 复制基准现金流
        stressed_cf = base_cash_flows.copy()
        
        # 应用收入冲击（逐渐加深）
        for t in range(months):
            shock_factor = shocks["revenue_shock"] * min(1, t / 12)  # 第一年内逐渐加深
            stressed_cf[t] += stressed_cf[t] * shock_factor
        
        # 应用成本增加
        for t in range(months):
            cost_factor = shocks["cost_increase"] * min(1, t / 6)  # 前6个月内逐渐加深
            stressed_cf[t] -= base_cash_flows[t] * cost_factor
        
        # 应用利率冲击（如果有利率路径）
        if interest_rate_paths is not None:
            for t in range(min(months, len(interest_rate_paths))):
                interest_effect = interest_rate_paths[t] * shocks["interest_shock"] * base_cash_flows[t] * 0.2
                stressed_cf[t] -= interest_effect
        
        # 存储压力情景下的现金流
        stressed_cash_flows[scenario] = stressed_cf
        
        # 计算NPV
        adjusted_discount_rate = discount_rate
        if scenario != "基准情景" and interest_rate_paths is not None:
            # 调整折现率
            adjusted_discount_rate += shocks["interest_shock"]
        
        # 计算月度折现率
        monthly_discount_rate = (1 + adjusted_discount_rate) ** (1/12) - 1
        
        # 计算NPV
        npv = 0
        for t in range(months):
            npv += stressed_cf[t] / (1 + monthly_discount_rate) ** t
        
        npv_results[scenario] = npv
    
    # 计算压力测试指标
    base_npv = npv_results["基准情景"]
    expected_loss = base_npv - np.mean([npv_results[s] for s in scenarios if s != "基准情景"])
    var_95 = base_npv - np.percentile([npv_results[s] for s in scenarios], 5)
    var_99 = base_npv - np.percentile([npv_results[s] for s in scenarios], 1)
    
    # 计算最大回撤
    cumulative_base = np.cumsum(base_cash_flows)
    worst_scenario = min(scenarios.keys(), key=lambda s: npv_results[s])
    cumulative_worst = np.cumsum(stressed_cash_flows[worst_scenario])
    drawdowns = [cumulative_base[t] - cumulative_worst[t] for t in range(months)]
    max_drawdown = max(drawdowns) if drawdowns else 0
    
    # 计算恢复概率
    recovery_months = 0
    for scenario in scenarios:
        if scenario != "基准情景":
            cf = stressed_cash_flows[scenario]
            for t in range(1, months):
                if cf[t] >= 0 and cf[t-1] < 0:
                    recovery_months += 1
                    break
    
    recovery_probability = recovery_months / (len(scenarios) - 1) if len(scenarios) > 1 else 0
    
    # 可视化压力测试结果
    plt.figure(figsize=(12, 8))
    
    # 绘制累积现金流对比
    plt.subplot(2, 1, 1)
    for scenario in scenarios:
        cumulative_cf = np.cumsum(stressed_cash_flows[scenario])
        plt.plot(cumulative_cf, label=scenario)
    
    plt.title('压力测试：累积现金流对比', fontsize=14)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('累积现金流 (万元)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制NPV对比
    plt.subplot(2, 1, 2)
    scenarios_list = list(scenarios.keys())
    npv_values = [npv_results[s] for s in scenarios_list]
    
    plt.bar(scenarios_list, npv_values, color=['blue', 'green', 'orange', 'red'])
    plt.title('压力测试：NPV对比', fontsize=14)
    plt.xlabel('情景', fontsize=12)
    plt.ylabel('NPV (万元)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'stress_test_{infra_type.value}.png'))
    plt.close()
    
    return {
        "stressed_cash_flows": stressed_cash_flows,
        "npv_results": npv_results,
        "expected_loss": expected_loss,
        "var_95": var_95,
        "var_99": var_99,
        "max_drawdown": max_drawdown,
        "recovery_probability": recovery_probability
    }

# 收入稳定性评估函数
def assess_revenue_stability(revenues, time_periods):
    # 计算收入增长率
    growth_rates = np.diff(revenues) / revenues[:-1]
    growth_rates = np.append(growth_rates, growth_rates[-1])  # 补充最后一个时间点
    
    # 计算移动平均和移动标准差
    window = min(12, len(revenues) // 3)  # 使用12个月或数据长度的1/3作为窗口
    rolling_mean = np.convolve(revenues, np.ones(window)/window, mode='valid')
    rolling_mean = np.append(np.array([rolling_mean[0]] * (window-1)), rolling_mean)  # 补充开始的点
    
    rolling_std = []
    for i in range(len(revenues)):
        start = max(0, i - window + 1)
        rolling_std.append(np.std(revenues[start:i+1]))
    rolling_std = np.array(rolling_std)
    
    # 计算稳定性指标：1 - (标准差/均值)
    stability = 1 - rolling_std / rolling_mean
    stability = np.clip(stability, 0, 1)  # 限制在0-1之间
    
    # 计算趋势强度（使用简单线性回归）
    x = np.arange(len(revenues))
    slope, _, r_value, _, _ = stats.linregress(x, revenues)
    trend_strength = r_value ** 2  # R²作为趋势强度
    
    # 计算季节性强度（如果数据足够长）
    seasonality = 0
    if len(revenues) >= 24:  # 至少需要2年数据
        # 使用简单的自相关来检测季节性
        autocorr = np.correlate(revenues, revenues, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # 只取正半部分
        autocorr = autocorr / autocorr[0]  # 归一化
        
        # 检查12个月的自相关
        if len(autocorr) > 12:
            seasonality = autocorr[12]  # 12个月的自相关系数
    
    # 计算平均稳定性和标准差
    average_stability = np.mean(stability)
    stability_std = np.std(stability)
    
    # 计算年化波动率
    annual_volatility = np.std(growth_rates) * np.sqrt(12)  # 假设月度数据，年化
    
    # 计算收入增长率
    if len(revenues) > 1:
        total_growth = (revenues[-1] / revenues[0]) ** (12 / len(revenues)) - 1  # 年化增长率
    else:
        total_growth = 0
    
    # 可视化稳定性分析
    plt.figure(figsize=(12, 10))
    
    # 收入时间序列
    plt.subplot(3, 1, 1)
    plt.plot(time_periods[:len(revenues)], revenues, 'b-', label='实际收入')
    plt.plot(time_periods[:len(rolling_mean)], rolling_mean, 'r--', label=f'{window}个月移动平均')
    plt.title('收入时间序列与移动平均', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('收入 (万元)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 稳定性指标
    plt.subplot(3, 1, 2)
    plt.plot(time_periods[:len(stability)], stability, 'g-')
    plt.axhline(y=average_stability, color='r', linestyle='--', label=f'平均稳定性: {average_stability:.4f}')
    plt.title('收入稳定性指标', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('稳定性 (0-1)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 收入增长率
    plt.subplot(3, 1, 3)
    plt.plot(time_periods[:len(growth_rates)], growth_rates, 'orange')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title(f'收入增长率 (年化: {total_growth:.2%}, 波动率: {annual_volatility:.2%})', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('月度增长率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'revenue_stability.png'))
    plt.close()
    
    return {
        "stability": stability,
        "average_stability": average_stability,
        "stability_std": stability_std,
        "trend_strength": trend_strength,
        "seasonality": seasonality,
        "annual_volatility": annual_volatility,
        "revenue_growth": total_growth
    }

# 数据加载函数
def load_data(file_path, sheet_name=None):
    try:
        if sheet_name:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            data = pd.read_excel(file_path)
        
        # 检查数据是否包含时间列
        if '时间' in data.columns:
            # 尝试不同的时间格式
            try:
                data['时间'] = pd.to_datetime(data['时间'])
            except:
                try:
                    data['时间'] = pd.to_datetime(data['时间'], format='%Y-%m')
                except:
                    try:
                        data['时间'] = pd.to_datetime(data['时间'], format='%Y/%m')
                    except:
                        print(f"警告：无法解析时间列格式，保持原样。")
        
        return data
    except Exception as e:
        print(f"加载数据时出错：{e}")
        return None

# 数据清洗和插值函数
def clean_and_interpolate(data, target_dates=None):
    if data is None or len(data) == 0:
        return None
    
    # 复制数据以避免修改原始数据
    df = data.copy()
    
    # 确保时间列是日期时间类型
    if '时间' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['时间']):
        try:
            df['时间'] = pd.to_datetime(df['时间'])
        except:
            print("警告：无法将时间列转换为日期时间类型。")
            return None
    
    # 处理缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 使用线性插值填充缺失值
        if df[col].isnull().any():
            df[col] = df[col].interpolate(method='linear')
    
    # 如果提供了目标日期，则重新采样到这些日期
    if target_dates is not None and '时间' in df.columns:
        # 设置时间为索引
        df.set_index('时间', inplace=True)
        
        # 创建一个新的DataFrame，包含目标日期
        new_df = pd.DataFrame(index=target_dates)
        
        # 对每一列进行重采样和插值
        for col in df.columns:
            # 创建插值函数
            f = interp1d(df.index.astype(np.int64), df[col].values, 
                         kind='linear', bounds_error=False, fill_value='extrapolate')
            
            # 应用插值函数到目标日期
            new_df[col] = f(target_dates.astype(np.int64))
        
        # 重置索引，将时间作为列
        new_df.reset_index(inplace=True)
        new_df.rename(columns={'index': '时间'}, inplace=True)
        
        return new_df
    
    return df

# 随机森林模型预测经济指标
def random_forest_forecast(data, target_cols, forecast_periods=120, test_size=0.2):
    if data is None or len(data) < 10:  # 确保有足够的数据进行训练
        print("数据不足，无法进行随机森林预测。")
        return None
    
    # 复制数据
    df = data.copy()
    
    # 确保时间列是日期时间类型
    if '时间' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['时间']):
            try:
                df['时间'] = pd.to_datetime(df['时间'])
            except:
                print("警告：无法将时间列转换为日期时间类型。")
                return None
    else:
        print("数据中缺少时间列。")
        return None
    
    # 创建特征
    df = df.sort_values('时间')
    df['月份'] = df['时间'].dt.month
    df['年份'] = df['时间'].dt.year
    df['季度'] = df['时间'].dt.quarter
    
    # 为每个目标列创建滞后特征
    for col in target_cols:
        if col in df.columns:
            for lag in range(1, 13):  # 创建1-12个月的滞后
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # 删除包含NaN的行
    df = df.dropna()
    
    # 准备特征和目标
    feature_cols = ['月份', '年份', '季度'] + [col for col in df.columns if '_lag' in col]
    X = df[feature_cols]
    
    # 存储预测结果
    forecasts = {}
    
    # 对每个目标列进行预测
    for col in target_cols:
        if col in df.columns:
            y = df[col]
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            # 训练随机森林模型
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # 评估模型
            y_pred = rf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"{col} 预测性能 - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # 准备预测数据
            last_date = df['时间'].max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
            
            # 创建预测DataFrame
            future_df = pd.DataFrame({'时间': future_dates})
            future_df['月份'] = future_df['时间'].dt.month
            future_df['年份'] = future_df['时间'].dt.year
            future_df['季度'] = future_df['时间'].dt.quarter
            
            # 初始化滞后值（使用最后已知的值）
            last_values = df[col].tail(12).values
            for i, lag in enumerate(range(1, 13)):
                if i < len(last_values):
                    future_df[f'{col}_lag{lag}'] = last_values[-(i+1)]
                else:
                    future_df[f'{col}_lag{lag}'] = last_values[0]
            
            # 逐步预测
            predictions = []
            for i in range(forecast_periods):
                # 准备当前时间点的特征
                current_features = future_df.iloc[i:i+1][feature_cols]
                
                # 预测
                pred = rf.predict(current_features)[0]
                predictions.append(pred)
                
                # 更新滞后特征
                if i + 1 < forecast_periods:
                    for lag in range(12, 1, -1):
                        future_df.loc[i+1, f'{col}_lag{lag}'] = future_df.loc[i, f'{col}_lag{lag-1}']
                    future_df.loc[i+1, f'{col}_lag1'] = pred
            
            # 存储预测结果
            future_df[col] = predictions
            forecasts[col] = future_df[['时间', col]]
    
    # 合并所有预测结果
    if forecasts:
        result = forecasts[list(forecasts.keys())[0]][['时间']]
        for col, forecast in forecasts.items():
            result[col] = forecast[col].values
        
        # 可视化预测结果
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(forecasts.keys()):
            plt.subplot(len(forecasts), 1, i+1)
            
            # 绘制历史数据
            plt.plot(df['时间'], df[col], 'b-', label='历史数据')
            
            # 绘制预测数据
            plt.plot(result['时间'], result[col], 'r--', label='预测')
            
            plt.title(f'{col} 预测', fontsize=14)
            plt.xlabel('时间', fontsize=12)
            plt.ylabel(col, fontsize=12)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'rf_forecast.png'))
        plt.close()
        
        return result
    
    return None