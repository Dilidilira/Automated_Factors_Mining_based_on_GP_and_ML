import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 读取因子重要性
df_imp = pd.read_csv('trees/Factors_selection.csv')
# 读取IC相关数据
df_ic = pd.read_csv('results/alphas191_ic_stats.csv')

# 合并，按 Alpha_index/Features 对齐
df = pd.merge(df_imp, df_ic, left_on='Features', right_on='Alpha_index', how='left')

df['IC mean abs'] = abs(df['IC mean'])

# 选择TOPSIS评价的列
cols = ['XGBoost_Importance', 'LightGBM_Importance', 'CatBoost_Importance', 'IC mean abs', 'IR']
weights = np.array([1, 1, 1, 1, 1])  # 可调整权重

def topsis(df, weights):
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(df)
    weighted = norm_data * weights
    ideal = np.max(weighted, axis=0)
    anti_ideal = np.min(weighted, axis=0)
    dist_ideal = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    dist_anti_ideal = np.sqrt(np.sum((weighted - anti_ideal) ** 2, axis=1))
    score = dist_anti_ideal / (dist_ideal + dist_anti_ideal)
    return score

df_topsis = df[cols]
df['topsis_score'] = topsis(df_topsis, weights)

# 按得分排序
df_sorted = df.sort_values('topsis_score', ascending=False)
print(df_sorted[['Features', 'topsis_score']])

# 保存结果
df_sorted.to_csv('results/Factors_topsis_ranked.csv', index=False)