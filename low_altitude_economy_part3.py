# CIR模型实现
def cir_model(data, forecast_data, simulation_periods=120, num_paths=1000, dt=1/12):
    # 提取历史利率数据
    if '利率' in data.columns:
        rates = data['利率'].values
    else:
        print("数据中缺少利率列，使用SHIBOR 3M数据代替。")
        if 'SHIBOR_3M' in data.columns:
            rates = data['SHIBOR_3M'].values / 100  # 转换为小数
        else:
            print("数据中缺少SHIBOR_3M列，使用默认值。")
            rates = np.array([0.025] * len(data))  # 默认值2.5%
    
    # 确保利率为正
    rates = np.maximum(rates, 0.001)  # 设置最小值为0.1%
    
    # 估计CIR模型参数
    def objective(params):
        kappa, theta, sigma = params
        if kappa <= 0 or theta <= 0 or sigma <= 0:
            return 1e10  # 惩罚负值
        
        # 计算对数似然
        log_likelihood = 0
        for i in range(1, len(rates)):
            mu = rates[i-1] + kappa * (theta - rates[i-1]) * dt
            var = sigma**2 * rates[i-1] * dt
            log_likelihood -= -0.5 * ((rates[i] - mu)**2 / var + np.log(2 * np.pi * var))
        
        return -log_likelihood
    
    # 初始参数猜测
    initial_guess = [0.5, np.mean(rates), 0.05]
    
    # 参数约束
    bounds = [(0.01, 10), (0.001, 0.2), (0.01, 0.5)]
    
    # 优化
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    kappa, theta, sigma = result.x
    
    print(f"CIR模型参数估计：kappa={kappa:.4f}, theta={theta:.4f}, sigma={sigma:.4f}")
    
    # 检查Feller条件
    feller_condition = 2 * kappa * theta > sigma**2
    if not feller_condition:
        print("警告：Feller条件不满足，利率可能达到零。调整参数...")
        # 调整参数以满足Feller条件
        sigma = np.sqrt(2 * kappa * theta * 0.9)  # 设置为略小于临界值
        print(f"调整后的sigma={sigma:.4f}")
    
    # 基于经济预测动态调整theta
    dynamic_theta = np.ones(simulation_periods) * theta
    
    # 如果有经济预测数据，调整theta
    if forecast_data is not None:
        # 检查是否有CPI预测
        if 'CPI' in forecast_data.columns:
            cpi_forecast = forecast_data['CPI'].values
            # 根据CPI调整theta（通胀上升，利率上升）
            for i in range(min(len(cpi_forecast), simulation_periods)):
                # 计算CPI同比变化率
                if i >= 12 and i < len(cpi_forecast):
                    cpi_yoy = (cpi_forecast[i] / cpi_forecast[i-12] - 1) * 100
                    # 当CPI同比增长超过3%，上调theta
                    if cpi_yoy > 3:
                        dynamic_theta[i] = theta * (1 + (cpi_yoy - 3) * 0.05)
                    # 当CPI同比增长低于1%，下调theta
                    elif cpi_yoy < 1:
                        dynamic_theta[i] = theta * (1 - (1 - cpi_yoy) * 0.05)
        
        # 检查是否有GDP预测
        if 'GDP' in forecast_data.columns:
            gdp_forecast = forecast_data['GDP'].values
            # 根据GDP调整theta（GDP增长强劲，利率上升）
            for i in range(min(len(gdp_forecast), simulation_periods)):
                # 计算GDP同比变化率
                if i >= 12 and i < len(gdp_forecast):
                    gdp_yoy = (gdp_forecast[i] / gdp_forecast[i-12] - 1) * 100
                    # 当GDP同比增长超过6%，上调theta
                    if gdp_yoy > 6:
                        dynamic_theta[i] = dynamic_theta[i] * (1 + (gdp_yoy - 6) * 0.02)
                    # 当GDP同比增长低于3%，下调theta
                    elif gdp_yoy < 3:
                        dynamic_theta[i] = dynamic_theta[i] * (1 - (3 - gdp_yoy) * 0.02)
    
    # 限制theta的变化范围
    dynamic_theta = np.clip(dynamic_theta, theta * 0.5, theta * 2.0)
    
    # 模拟利率路径
    np.random.seed(42)
    r0 = rates[-1]  # 使用最后一个观测值作为初始值
    
    # 初始化路径数组
    paths = np.zeros((num_paths, simulation_periods + 1))
    paths[:, 0] = r0
    
    # 添加随机波动性
    random_volatility = np.random.uniform(0.8, 1.2, num_paths)
    
    # 添加跳跃扩散
    jump_intensity = 0.05  # 每年跳跃概率
    jump_size_mean = 0.002  # 平均跳跃大小
    jump_size_std = 0.001  # 跳跃大小标准差
    
    # 模拟路径
    for i in range(num_paths):
        r = r0
        for t in range(simulation_periods):
            # 应用动态theta
            current_theta = dynamic_theta[t]
            
            # 添加随机波动性
            current_sigma = sigma * random_volatility[i]
            
            # CIR过程
            drift = kappa * (current_theta - r) * dt
            diffusion = current_sigma * np.sqrt(r * dt) * np.random.normal(0, 1)
            
            # 添加跳跃（泊松过程）
            jump = 0
            if np.random.random() < jump_intensity * dt:
                jump = np.random.normal(jump_size_mean, jump_size_std)
                # 跳跃方向（上升或下降）
                if np.random.random() < 0.5:
                    jump = -jump
            
            # 更新利率
            r = r + drift + diffusion + jump
            
            # 确保利率为正
            r = max(r, 0.001)
            
            paths[i, t+1] = r
    
    # 计算平均路径和分位数
    mean_path = np.mean(paths, axis=0)
    percentile_5 = np.percentile(paths, 5, axis=0)
    percentile_95 = np.percentile(paths, 95, axis=0)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 绘制历史数据
    time_points = np.arange(len(rates))
    plt.plot(time_points, rates, 'b-', label='历史利率')
    
    # 绘制预测路径
    forecast_time = np.arange(len(rates) - 1, len(rates) + simulation_periods)
    plt.plot(forecast_time, mean_path, 'r-', label='平均预测路径')
    plt.fill_between(forecast_time, percentile_5, percentile_95, color='r', alpha=0.2, label='90%置信区间')
    
    # 绘制一些样本路径
    for i in range(min(10, num_paths)):
        plt.plot(forecast_time, paths[i], 'r-', alpha=0.1)
    
    plt.title('CIR模型利率预测', fontsize=14)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('利率', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cir_model_forecast.png'))
    plt.close()
    
    # 返回模型参数和预测路径
    return {
        "parameters": {"kappa": kappa, "theta": theta, "sigma": sigma},
        "dynamic_theta": dynamic_theta,
        "paths": paths,
        "mean_path": mean_path,
        "percentile_5": percentile_5,
        "percentile_95": percentile_95
    }

# Hull-White模型实现
def hull_white_model(data, forecast_data, simulation_periods=120, num_paths=1000, dt=1/12):
    # 提取历史利率数据
    if '利率' in data.columns:
        rates = data['利率'].values
    else:
        print("数据中缺少利率列，使用SHIBOR 3M数据代替。")
        if 'SHIBOR_3M' in data.columns:
            rates = data['SHIBOR_3M'].values / 100  # 转换为小数
        else:
            print("数据中缺少SHIBOR_3M列，使用默认值。")
            rates = np.array([0.025] * len(data))  # 默认值2.5%
    
    # 估计Hull-White模型参数
    def objective(params):
        alpha, sigma = params
        if alpha <= 0 or sigma <= 0:
            return 1e10  # 惩罚负值
        
        # 计算对数似然
        log_likelihood = 0
        for i in range(1, len(rates)):
            # 简化的Hull-White模型，假设theta(t)为常数
            theta = np.mean(rates)
            mu = rates[i-1] + alpha * (theta - rates[i-1]) * dt
            var = sigma**2 * dt
            log_likelihood -= -0.5 * ((rates[i] - mu)**2 / var + np.log(2 * np.pi * var))
        
        return -log_likelihood
    
    # 初始参数猜测
    initial_guess = [0.1, 0.01]
    
    # 参数约束
    bounds = [(0.01, 5), (0.001, 0.1)]
    
    # 优化
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    alpha, sigma = result.x
    
    print(f"Hull-White模型参数估计：alpha={alpha:.4f}, sigma={sigma:.4f}")
    
    # 估计初始theta值（长期均值）
    theta_0 = np.mean(rates)
    
    # 基于经济预测动态调整theta
    dynamic_theta = np.ones(simulation_periods) * theta_0
    
    # 如果有经济预测数据，调整theta
    if forecast_data is not None:
        # 检查是否有CPI预测
        if 'CPI' in forecast_data.columns:
            cpi_forecast = forecast_data['CPI'].values
            # 根据CPI调整theta（通胀上升，利率上升）
            for i in range(min(len(cpi_forecast), simulation_periods)):
                # 计算CPI同比变化率
                if i >= 12 and i < len(cpi_forecast):
                    cpi_yoy = (cpi_forecast[i] / cpi_forecast[i-12] - 1) * 100
                    # 当CPI同比增长超过3%，上调theta
                    if cpi_yoy > 3:
                        dynamic_theta[i] = theta_0 * (1 + (cpi_yoy - 3) * 0.05)
                    # 当CPI同比增长低于1%，下调theta
                    elif cpi_yoy < 1:
                        dynamic_theta[i] = theta_0 * (1 - (1 - cpi_yoy) * 0.05)
        
        # 检查是否有GDP预测
        if 'GDP' in forecast_data.columns:
            gdp_forecast = forecast_data['GDP'].values
            # 根据GDP调整theta（GDP增长强劲，利率上升）
            for i in range(min(len(gdp_forecast), simulation_periods)):
                # 计算GDP同比变化率
                if i >= 12 and i < len(gdp_forecast):
                    gdp_yoy = (gdp_forecast[i] / gdp_forecast[i-12] - 1) * 100
                    # 当GDP同比增长超过6%，上调theta
                    if gdp_yoy > 6:
                        dynamic_theta[i] = dynamic_theta[i] * (1 + (gdp_yoy - 6) * 0.02)
                    # 当GDP同比增长低于3%，下调theta
                    elif gdp_yoy < 3:
                        dynamic_theta[i] = dynamic_theta[i] * (1 - (3 - gdp_yoy) * 0.02)
        
        # 检查是否有失业率预测
        if '失业率' in forecast_data.columns:
            unemployment_forecast = forecast_data['失业率'].values
            # 根据失业率调整theta（失业率上升，利率下降）
            for i in range(min(len(unemployment_forecast), simulation_periods)):
                if i >= 12 and i < len(unemployment_forecast):
                    # 失业率变化
                    unemp_change = unemployment_forecast[i] - unemployment_forecast[i-12]
                    # 当失业率上升，下调theta
                    if unemp_change > 0.5:
                        dynamic_theta[i] = dynamic_theta[i] * (1 - unemp_change * 0.05)
                    # 当失业率下降，上调theta
                    elif unemp_change < -0.5:
                        dynamic_theta[i] = dynamic_theta[i] * (1 + abs(unemp_change) * 0.05)
    
    # 限制theta的变化范围
    dynamic_theta = np.clip(dynamic_theta, theta_0 * 0.5, theta_0 * 2.0)
    
    # 模拟利率路径
    np.random.seed(43)  # 不同于CIR模型的种子
    r0 = rates[-1]  # 使用最后一个观测值作为初始值
    
    # 初始化路径数组
    paths = np.zeros((num_paths, simulation_periods + 1))
    paths[:, 0] = r0
    
    # 模拟路径
    for i in range(num_paths):
        r = r0
        for t in range(simulation_periods):
            # 应用动态theta
            current_theta = dynamic_theta[t]
            
            # Hull-White过程
            drift = alpha * (current_theta - r) * dt
            diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1)
            
            # 更新利率
            r = r + drift + diffusion
            
            # 确保利率为正（虽然Hull-White模型允许负利率，但在这里我们限制为正）
            r = max(r, 0.001)
            
            paths[i, t+1] = r
    
    # 计算平均路径和分位数
    mean_path = np.mean(paths, axis=0)
    percentile_5 = np.percentile(paths, 5, axis=0)
    percentile_95 = np.percentile(paths, 95, axis=0)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 绘制历史数据
    time_points = np.arange(len(rates))
    plt.plot(time_points, rates, 'b-', label='历史利率')
    
    # 绘制预测路径
    forecast_time = np.arange(len(rates) - 1, len(rates) + simulation_periods)
    plt.plot(forecast_time, mean_path, 'g-', label='平均预测路径')
    plt.fill_between(forecast_time, percentile_5, percentile_95, color='g', alpha=0.2, label='90%置信区间')
    
    # 绘制一些样本路径
    for i in range(min(10, num_paths)):
        plt.plot(forecast_time, paths[i], 'g-', alpha=0.1)
    
    plt.title('Hull-White模型利率预测', fontsize=14)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('利率', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hull_white_model_forecast.png'))
    plt.close()
    
    # 返回模型参数和预测路径
    return {
        "parameters": {"alpha": alpha, "sigma": sigma, "theta_0": theta_0},
        "dynamic_theta": dynamic_theta,
        "paths": paths,
        "mean_path": mean_path,
        "percentile_5": percentile_5,
        "percentile_95": percentile_95
    }

# OAS模型实现
def oas_model(data, forecast_data, simulation_periods=120, num_paths=1000):
    # 提取历史利率和收益率数据
    if 'SHIBOR_3M' in data.columns and 'YIELD_5Y' in data.columns:
        short_rates = data['SHIBOR_3M'].values / 100  # 转换为小数
        bond_yields = data['YIELD_5Y'].values / 100  # 转换为小数
    else:
        print("数据中缺少SHIBOR_3M或YIELD_5Y列，使用默认值。")
        short_rates = np.array([0.025] * len(data))  # 默认值2.5%
        bond_yields = np.array([0.03] * len(data))  # 默认值3%
    
    # 计算历史OAS（简化计算：长期收益率 - 短期利率）
    historical_oas = bond_yields - short_rates
    
    # 估计OAS模型参数（假设OAS遵循均值回归过程）
    def objective(params):
        mean_oas, speed, vol = params
        if speed <= 0 or vol <= 0:
            return 1e10  # 惩罚负值
        
        # 计算对数似然
        log_likelihood = 0
        for i in range(1, len(historical_oas)):
            mu = historical_oas[i-1] + speed * (mean_oas - historical_oas[i-1]) * (1/12)
            var = vol**2 * (1/12)
            log_likelihood -= -0.5 * ((historical_oas[i] - mu)**2 / var + np.log(2 * np.pi * var))
        
        return -log_likelihood
    
    # 初始参数猜测
    initial_guess = [np.mean(historical_oas), 0.5, 0.01]
    
    # 参数约束
    bounds = [(0.001, 0.1), (0.01, 5), (0.001, 0.05)]
    
    # 优化
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    mean_oas, speed, vol = result.x
    
    print(f"OAS模型参数估计：mean_oas={mean_oas:.4f}, speed={speed:.4f}, vol={vol:.4f}")
    
    # 模拟未来OAS路径
    np.random.seed(44)  # 不同于前两个模型的种子
    oas_0 = historical_oas[-1]  # 使用最后一个观测值作为初始值
    
    # 初始化OAS路径数组
    oas_paths = np.zeros((num_paths, simulation_periods + 1))
    oas_paths[:, 0] = oas_0
    
    # 模拟OAS路径
    for i in range(num_paths):
        oas = oas_0
        for t in range(simulation_periods):
            # OAS均值回归过程
            drift = speed * (mean_oas - oas) * (1/12)
            diffusion = vol * np.sqrt(1/12) * np.random.normal(0, 1)
            
            # 更新OAS
            oas = oas + drift + diffusion
            
            # 确保OAS在合理范围内
            oas = max(oas, -0.01)  # 允许小幅负值
            oas = min(oas, 0.1)    # 上限10%
            
            oas_paths[i, t+1] = oas
    
    # 计算平均OAS路径和分位数
    mean_oas_path = np.mean(oas_paths, axis=0)
    percentile_5_oas = np.percentile(oas_paths, 5, axis=0)
    percentile_95_oas = np.percentile(oas_paths, 95, axis=0)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 绘制历史OAS
    time_points = np.arange(len(historical_oas))
    plt.plot(time_points, historical_oas, 'b-', label='历史OAS')
    
    # 绘制预测OAS路径
    forecast_time = np.arange(len(historical_oas) - 1, len(historical_oas) + simulation_periods)
    plt.plot(forecast_time, mean_oas_path, 'purple', label='平均预测OAS')
    plt.fill_between(forecast_time, percentile_5_oas, percentile_95_oas, color='purple', alpha=0.2, label='90%置信区间')
    
    # 绘制一些样本路径
    for i in range(min(10, num_paths)):
        plt.plot(forecast_time, oas_paths[i], 'purple', alpha=0.1)
    
    plt.title('OAS模型预测', fontsize=14)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('OAS', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'oas_model_forecast.png'))
    plt.close()
    
    # 返回模型参数和预测路径
    return {
        "parameters": {"mean_oas": mean_oas, "speed": speed, "vol": vol},
        "historical_oas": historical_oas,
        "oas_paths": oas_paths,
        "mean_oas_path": mean_oas_path,
        "percentile_5_oas": percentile_5_oas,
        "percentile_95_oas": percentile_95_oas
    }