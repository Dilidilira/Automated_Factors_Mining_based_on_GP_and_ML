import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os

frequency=60
factor_index=5
#################################################################

df=pd.read_csv(f'Factors/{frequency}m_Factor_{factor_index}.csv')
df['trade_time']=pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
df.set_index(df['trade_time'],inplace=True)

df_depolarized=pd.DataFrame()
df_standardized=pd.DataFrame()

Contracts = ['IF00','IF01','IF02','IH00','IH01','IH02','IC00','IC01','IC02','IM00','IM01','IM02']

##################################################################

#检查空值的个数
for contract in Contracts:
    if contract not in ['IM00','IM01','IM02']:
        print(f'{contract}因子长度：',len(df[contract]),f'{contract}空值个数：',len(df[contract])-len(df[contract].dropna()))
    else:
        print(f'{contract}因子长度：',len(df[df.index > '2022-7-21 15:00:00'][contract]),f'{contract}空值个数：',len(df[df.index > '2022-7-21 15:00:00'][contract])-len(df[df.index > '2022-7-21 15:00:00'][contract].dropna()))

#绘制因子分布散点图

    fig, axs = plt.subplots(1, 3, figsize=(18, 10))

    if contract not in ['IM00','IM01','IM02']:
        df['x'] = list(range(len(df[contract]))) #方便图表展示进行加入了标签列
        axs[0].scatter(df['x'], df[contract])
        axs[0].set_title(f'{contract} Raw Factor Scatter Plot')
        axs[0].set_xlabel('Index')
        axs[0].set_ylabel(contract)
        
    else:
        df1=df.copy()
        df1=df1[df1.index>'2022-7-21 15:00:00']
        df1['x'] = list(range(len(df1[contract])))
        axs[0].scatter(df1['x'], df1[contract])
        axs[0].set_title(f'{contract} Raw Factor Scatter Plot')
        axs[0].set_xlabel('Index')
        axs[0].set_ylabel(contract)

#绝对值中位数去极值法

    median = df[contract].median()
    mad = np.abs(df[contract] - median).median() # 计算绝对中位差
    
    # 设置上下限，超出该范围的值将被替换
    lower_bound = median - 3 * mad
    upper_bound = median + 3 * mad
    
    # 使用上下限对数据进行限制
    df_depolarized[contract]=df[contract].clip(lower_bound, upper_bound)
    
#绘制去极值后的散点图
    
    if contract not in ['IM00','IM01','IM02']:
        df_depolarized['x'] = list(range(len(df_depolarized[contract]))) 
        axs[1].scatter(df_depolarized['x'], df_depolarized[contract])
        axs[1].set_title(f'{contract} Depolarized Factor Scatter Plot')
        axs[1].set_xlabel('Index')
        axs[1].set_ylabel(contract)
    else:
        df_depolarized_1=df_depolarized.copy()
        df_depolarized_1=df_depolarized_1[df_depolarized_1.index>'2022-7-21 15:00:00']
        df_depolarized_1['x'] = list(range(len(df_depolarized_1[contract])))
        axs[1].scatter(df_depolarized_1['x'], df_depolarized_1[contract])
        axs[1].set_title(f'{contract} Depolarized Factor Scatter Plot')
        axs[1].set_xlabel('Index')
        axs[1].set_ylabel(contract)
        
#z_score标准化因子

    df_standardized[contract]=(df_depolarized[contract]-df_depolarized[contract].mean())/df_depolarized[contract].std()
    
    sns.histplot(df_standardized[contract], kde=True,ax=axs[2])
    axs[2].set_title('Standardized Distribution')
    axs[2].set_xlabel('Value')
    axs[2].set_ylabel('Frequency')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

df_standardized.to_csv(f'Factors/{frequency}m_Factor_{factor_index}_preprocessed.csv')










