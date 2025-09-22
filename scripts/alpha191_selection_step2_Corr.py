import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_heatmap(factor_indexes,start_datetime='2019-11-01 08:00:00',end_datetime='2024-11-01 15:00:00'):
    #导入alpha191因子数据
    df = pd.read_csv('/hdd/hdd1/cji/Trees/alphas191_data.csv')
    df['trade_time']=pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df=df.set_index('trade_time',drop=True)

    corr_matrix_list=[]

    grouped=df.groupby('level_1')
    for name, group in grouped:
        data=group.loc[:,factor_indexes]
        data_filtered=data[(data.index>=start_datetime)&(data.index<=end_datetime)]

        correlation_matrix=data_filtered.corr()
        corr_matrix_list.append(correlation_matrix)

    average_matrix=sum(corr_matrix_list)/len(corr_matrix_list)
    plt.figure(figsize=(12, 10))
    # 绘制热力图
    sns.heatmap(
        average_matrix,            # 相关性矩阵
        annot=True,                   # 是否显示数值
        cmap='coolwarm',               # 配色方案（冷暖色调）
        vmin=-1, vmax=1,               # 相关性范围（-1到1）
        cbar=True                      # 是否显示颜色条
    )
    plt.show()
    return average_matrix


def corr_filter(average_matrix,factor_indexes,threshold=0.8):
    high_corr_pairs = []

    for i in range(len(average_matrix.columns)):
        for j in range(i + 1, len(average_matrix.columns)):
            if average_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append((average_matrix.columns[i], average_matrix.columns[j], average_matrix.iloc[i, j]))

    high_corr_df = pd.DataFrame(high_corr_pairs, columns=["Factor1", "Factor2", "Correlation"])
    high_corr_df = high_corr_df.sort_values(by="Correlation", ascending=False)


    factors_in_high_corr = set(high_corr_df["Factor1"]).union(set(high_corr_df["Factor2"]))
    missing_factors = [factor for factor in factor_indexes if factor not in factors_in_high_corr]

    return high_corr_df,missing_factors

if __name__ == "__main__":

    #导入alpha191名称
    alphas191_index=pd.read_csv('/hdd/hdd1/cji/alphas191_index.csv')
    factor_indexes=alphas191_index['Alpha'].to_list()
    
    average_matrix=correlation_heatmap(factor_indexes)

    high_corr_df, missing_factors=corr_filter(average_matrix)

    # 打印或保存结果
    print("Independent Factors:", missing_factors)

    # 保存到文件
    pd.DataFrame(missing_factors, columns=["Missing_Factors"]).to_csv('/hdd/hdd1/cji/Trees/independent_factors.csv', index=False)
    high_corr_df.to_csv('/hdd/hdd1/cji/Trees/alphas191_corr.csv',index=False)










