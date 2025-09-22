from pulp import LpMaximize, LpProblem, LpVariable
import numpy as np
import pandas as pd

"""
冗余因子会影响树模型对因子重要性的判断，因此需要去除alpha191中的冗余因子
step2中设置threshold将相关性高于threshold的因子对判定为冗余，目标是在冗余的因子群中找到尽可能多的两两不相互冗余的因子群，
且寻找出的无冗余因子群中的因子自身的预测能力要尽可能的强

"""

#导入冗余因子群
high_corr_df= pd.read_csv('trees/alphas191_corr.csv')
factors_in_high_corr = set(high_corr_df["Factor1"]).union(set(high_corr_df["Factor2"]))

#创建最大化问题
problem = LpProblem("Maximize_Score", LpMaximize)

#定义决策变量: factor=1 代表被选入因子池
variables = {factor: LpVariable(factor, 0, 1, cat="Binary") for factor in factors_in_high_corr}

#设置因子群因子权重
weights={}#储存冗余因子群的ICmean值作为权重

stats=pd.read_csv('results/alphas191_ic_stats.csv')
for factor in factors_in_high_corr:
    weight=stats[stats['Alpha_index']==factor]['IC mean'].iloc[0]
    weight=abs(weight)
    weights[factor]=weight

#定义目标函数：最大化因子池中IC_mean的绝对值之和
problem += sum(weights[factor] * variables[factor] for factor in factors_in_high_corr), "Total_Score"

#添加约束条件
grouped=high_corr_df.groupby('Factor1')

for name, group in grouped:
    print(f"正在处理{name}的约束:")
    related_factors=group['Factor2'].to_list()
    print(f"与{name}高度相关的因子个数为：{len(related_factors)-1}")
    for related_factor in related_factors:
        problem += sum([variables[name],variables[related_factor]]) <= 1

#求解
problem.solve()

#输出结果
print("Status:", problem.status)
print("最佳解:")

factors_in_low_corr=[]
for var_name, var in variables.items():
    print(f"{var_name}: {int(var.value())}")
    if int(var.value())==1:
        factors_in_low_corr.append(var_name)
print("最大得分:", problem.objective.value())
print(f"冗余因子群个数：{len(factors_in_high_corr)},提取的非冗余因子个数：{len(factors_in_low_corr)}")


#导入原非冗余因子群

non_redundant_factors=pd.read_csv('trees/independent_factors.csv')
non_redundant_factors=non_redundant_factors['Missing_Factors'].to_list()

#加和新非冗余因子群
non_redundant_factors=non_redundant_factors+factors_in_low_corr


#检查新非冗余因子群是否存在冗余因子
#average_matrix=correlation_heatmap(non_redundant_factors)
#high_corr_df,missing_factors=corr_filter(average_matrix,non_redundant_factors)

#print(high_corr_df)


pd.DataFrame(non_redundant_factors, columns=["Factor_Index"]).to_csv('trees/Non_redundant_factors.csv', index=False)

