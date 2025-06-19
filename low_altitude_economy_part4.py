# 蒙特卡洛模拟和资产证券化分析
def monte_carlo_securitization(infra_type, revenue_type, data, forecast_data, simulation_periods=120, num_paths=1000):
    print(f"\n开始对{infra_type.value}（{revenue_type.value}）进行蒙特卡洛模拟和资产证券化分析...")
    
    # 准备数据
    # 提取经济指标数据
    cpi_data = None
    gdp_data = None
    unemployment_data = None
    if 'CPI' in data.columns:
        cpi_data = data['CPI'].values
    if 'GDP' in data.columns:
        gdp_data = data['GDP'].values
    if '失业率' in data.columns:
        unemployment_data = data['失业率'].values
    
    # 提取利率数据
    if 'SHIBOR_3M' in data.columns:
        rate_data = data['SHIBOR_3M'].values / 100  # 转换为小数
    else:
        rate_data = np.array([0.025] * len(data))  # 默认值2.5%
    
    # 估计CIR模型参数
    print("估计CIR模型参数...")
    cir_result = cir_model(data, forecast_data, simulation_periods, num_paths)
    
    # 估计Hull-White模型参数
    print("估计Hull-White模型参数...")
    hw_result = hull_white_model(data, forecast_data, simulation_periods, num_paths)
    
    # 模拟利率路径（使用两种模型的平均）
    print("模拟利率路径...")
    rate_paths = (cir_result['paths'] + hw_result['paths']) / 2
    
    # 初始化现金流数组
    cash_flows = np.zeros((num_paths, simulation_periods))
    
    # 模拟现金流
    print("模拟现金流...")
    for i in range(num_paths):
        # 获取当前路径的利率
        current_rates = rate_paths[i, 1:]  # 跳过初始值
        
        # 模拟现金流
        cf = simulate_cash_flows(infra_type, revenue_type, simulation_periods, current_rates, 
                                cpi_data, gdp_data, unemployment_data)
        
        cash_flows[i] = cf
    
    # 计算NPV
    print("计算NPV...")
    npv_values = np.zeros(num_paths)
    for i in range(num_paths):
        # 使用当前路径的利率作为折现率
        discount_factors = np.array([1 / ((1 + rate_paths[i, t]) ** (t/12)) for t in range(1, simulation_periods + 1)])
        npv_values[i] = np.sum(cash_flows[i] * discount_factors)
    
    # 计算NPV统计量
    mean_npv = np.mean(npv_values)
    median_npv = np.median(npv_values)
    std_npv = np.std(npv_values)
    percentile_5 = np.percentile(npv_values, 5)
    percentile_95 = np.percentile(npv_values, 95)
    
    # 可视化NPV分布
    plt.figure(figsize=(12, 10))
    
    # NPV分布直方图
    plt.subplot(2, 1, 1)
    plt.hist(npv_values, bins=50, alpha=0.7, color='blue')
    plt.axvline(mean_npv, color='red', linestyle='--', label=f'平均值: {mean_npv:.2f}万元')
    plt.axvline(median_npv, color='green', linestyle='--', label=f'中位数: {median_npv:.2f}万元')
    plt.axvline(percentile_5, color='orange', linestyle='--', label=f'5%分位数: {percentile_5:.2f}万元')
    plt.axvline(percentile_95, color='purple', linestyle='--', label=f'95%分位数: {percentile_95:.2f}万元')
    
    plt.title(f'{infra_type.value}（{revenue_type.value}）NPV分布', fontsize=14)
    plt.xlabel('NPV (万元)', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 现金流预测
    plt.subplot(2, 1, 2)
    mean_cash_flows = np.mean(cash_flows, axis=0)
    percentile_5_cf = np.percentile(cash_flows, 5, axis=0)
    percentile_95_cf = np.percentile(cash_flows, 95, axis=0)
    
    time_points = np.arange(simulation_periods)
    plt.plot(time_points, mean_cash_flows, 'b-', label='平均现金流')
    plt.fill_between(time_points, percentile_5_cf, percentile_95_cf, color='blue', alpha=0.2, label='90%置信区间')
    
    plt.title(f'{infra_type.value}（{revenue_type.value}）现金流预测', fontsize=14)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('现金流 (万元)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'mc_securitization_{infra_type.value}_{revenue_type.value}.png'))
    plt.close()
    
    # 保存结果到Excel
    result_df = pd.DataFrame({
        '统计量': ['平均NPV', '中位数NPV', 'NPV标准差', '5%分位数', '95%分位数'],
        '值 (万元)': [mean_npv, median_npv, std_npv, percentile_5, percentile_95]
    })
    
    # 保存现金流预测
    cf_df = pd.DataFrame({
        '月份': time_points,
        '平均现金流': mean_cash_flows,
        '5%分位数': percentile_5_cf,
        '95%分位数': percentile_95_cf
    })
    
    # 创建Excel写入器
    excel_path = os.path.join(OUTPUT_DIR, f'mc_securitization_{infra_type.value}_{revenue_type.value}.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        result_df.to_excel(writer, sheet_name='NPV统计', index=False)
        cf_df.to_excel(writer, sheet_name='现金流预测', index=False)
    
    print(f"{infra_type.value}（{revenue_type.value}）蒙特卡洛模拟和资产证券化分析完成。")
    
    return {
        "npv_values": npv_values,
        "cash_flows": cash_flows,
        "mean_npv": mean_npv,
        "median_npv": median_npv,
        "std_npv": std_npv,
        "percentile_5": percentile_5,
        "percentile_95": percentile_95,
        "mean_cash_flows": mean_cash_flows
    }

# 辅助函数：CIR路径模拟
def simulate_cir_path(r0, kappa, theta, sigma, periods, dt=1/12):
    path = np.zeros(periods + 1)
    path[0] = r0
    
    for t in range(periods):
        drift = kappa * (theta - path[t]) * dt
        diffusion = sigma * np.sqrt(path[t] * dt) * np.random.normal(0, 1)
        path[t+1] = path[t] + drift + diffusion
        path[t+1] = max(path[t+1], 0.001)  # 确保利率为正
    
    return path

# 辅助函数：Hull-White路径模拟
def simulate_hw_path(r0, alpha, theta, sigma, periods, dt=1/12):
    path = np.zeros(periods + 1)
    path[0] = r0
    
    for t in range(periods):
        drift = alpha * (theta - path[t]) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1)
        path[t+1] = path[t] + drift + diffusion
        path[t+1] = max(path[t+1], 0.001)  # 确保利率为正
    
    return path

# 加载数据
def load_all_data():
    # 加载CPI数据
    cpi_data = load_data('data/economic_indicators.xlsx', 'CPI')
    if cpi_data is not None:
        print("CPI数据加载成功。")
    else:
        print("警告：CPI数据加载失败。")
    
    # 加载货币供应量数据
    money_data = load_data('data/economic_indicators.xlsx', 'Money')
    if money_data is not None:
        print("货币供应量数据加载成功。")
    else:
        print("警告：货币供应量数据加载失败。")
    
    # 加载失业率数据
    unemployment_data = load_data('data/economic_indicators.xlsx', 'Unemployment')
    if unemployment_data is not None:
        print("失业率数据加载成功。")
    else:
        print("警告：失业率数据加载失败。")
    
    # 加载GDP数据
    gdp_data = load_data('data/economic_indicators.xlsx', 'GDP')
    if gdp_data is not None:
        print("GDP数据加载成功。")
    else:
        print("警告：GDP数据加载失败。")
    
    # 加载房地产数据
    housing_data = load_data('data/economic_indicators.xlsx', 'Housing')
    if housing_data is not None:
        print("房地产数据加载成功。")
    else:
        print("警告：房地产数据加载失败。")
    
    # 加载SHIBOR数据
    shibor_data = load_data('data/interest_rates.xlsx', 'SHIBOR')
    if shibor_data is not None:
        print("SHIBOR数据加载成功。")
    else:
        print("警告：SHIBOR数据加载失败。")
    
    # 加载收益率数据
    yield_data = load_data('data/interest_rates.xlsx', 'YIELD')
    if yield_data is not None:
        print("收益率数据加载成功。")
    else:
        print("警告：收益率数据加载失败。")
    
    # 合并所有数据
    # 首先确定共同的时间范围
    all_dfs = [df for df in [cpi_data, money_data, unemployment_data, gdp_data, housing_data, shibor_data, yield_data] if df is not None]
    
    if not all_dfs:
        print("错误：没有成功加载任何数据。")
        return None
    
    # 合并数据
    merged_data = all_dfs[0]
    for df in all_dfs[1:]:
        merged_data = pd.merge(merged_data, df, on='时间', how='outer')
    
    # 按时间排序
    merged_data = merged_data.sort_values('时间')
    
    # 清洗和插值
    cleaned_data = clean_and_interpolate(merged_data)
    
    return cleaned_data

# 运行所有模型
def run_all_models(data, simulation_periods=120, num_paths=1000):
    # 随机森林预测经济指标
    print("\n使用随机森林预测经济指标...")
    forecast_data = random_forest_forecast(data, ['CPI', 'M2', '失业率', 'GDP'], simulation_periods)
    
    # CIR模型
    print("\n运行CIR模型...")
    cir_result = cir_model(data, forecast_data, simulation_periods, num_paths)
    
    # Hull-White模型
    print("\n运行Hull-White模型...")
    hw_result = hull_white_model(data, forecast_data, simulation_periods, num_paths)
    
    # OAS模型
    print("\n运行OAS模型...")
    oas_result = oas_model(data, forecast_data, simulation_periods, num_paths)
    
    return {
        "forecast_data": forecast_data,
        "cir_result": cir_result,
        "hw_result": hw_result,
        "oas_result": oas_result
    }

# 主函数
def main():
    # 加载数据
    print("加载经济和利率数据...")
    data = load_all_data()
    
    if data is None:
        print("错误：数据加载失败，程序终止。")
        return
    
    # 运行所有模型
    print("运行所有模型...")
    model_results = run_all_models(data)
    
    # 获取预测数据和模型结果
    forecast_data = model_results["forecast_data"]
    cir_result = model_results["cir_result"]
    hw_result = model_results["hw_result"]
    oas_result = model_results["oas_result"]
    
    # 对每种基础设施类型进行分析
    infrastructure_results = {}
    
    for infra_type in InfrastructureType:
        print(f"\n分析{infra_type.value}基础设施...")
        
        # 对每种收入类型进行分析
        revenue_results = {}
        for revenue_type in RevenueType:
            print(f"分析{revenue_type.value}收入...")
            
            # 蒙特卡洛模拟和资产证券化分析
            mc_result = monte_carlo_securitization(infra_type, revenue_type, data, forecast_data)
            
            # 计算稳定性指标
            stability_result = assess_revenue_stability(mc_result["mean_cash_flows"], 
                                                     np.arange(len(mc_result["mean_cash_flows"])))
            
            # 风险评估
            risk_result = assess_risk(infra_type, forecast_data)
            
            # 压力测试
            stress_result = stress_test(infra_type, mc_result["mean_cash_flows"], 
                                       cir_result["mean_path"][1:], forecast_data)
            
            # 存储结果
            revenue_results[revenue_type] = {
                "mc_result": mc_result,
                "stability_result": stability_result,
                "risk_result": risk_result,
                "stress_result": stress_result
            }
        
        # 存储该基础设施类型的所有结果
        infrastructure_results[infra_type] = revenue_results
    
    # 计算总体NPV
    total_npv = 0
    for infra_type in InfrastructureType:
        for revenue_type in RevenueType:
            total_npv += infrastructure_results[infra_type][revenue_type]["mc_result"]["mean_npv"]
    
    print(f"\n总体NPV: {total_npv:.2f}万元")
    
    return infrastructure_results