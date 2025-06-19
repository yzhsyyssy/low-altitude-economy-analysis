# 低空经济资产证券化分析主程序

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# 导入各部分代码
from low_altitude_economy_part1 import *
from low_altitude_economy_part2 import *
from low_altitude_economy_part3 import *
from low_altitude_economy_part4 import *
from low_altitude_economy_part5 import *

# 设置警告过滤
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建输出目录
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("开始低空经济资产证券化分析...")
    main()
    print("分析完成！")
