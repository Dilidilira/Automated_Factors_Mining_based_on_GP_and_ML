import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os

#4/38/56/101/123/148/154为分类变量

frequencies = ['15m_','30m_','60m_']
alphas191_index=pd.read_csv('alphas191_index.csv')
factor_indexes=alphas191_index['Alpha'].to_list()

for frequency in frequencies:
    for factor_index in factor_indexes:

        df=pd.read_csv(f'191alpha_Factors/{frequency}{factor_index}.csv')
        df['trade_time']=pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
        df.set_index(df['trade_time'],inplace=True)

        df_depolarized=pd.DataFrame()
        df_standardized=pd.DataFrame()

        Contracts = ['IF00','IF01','IF02','IH00','IH01','IH02','IC00','IC01','IC02','IM00','IM01','IM02']

    #绝对值中位数去极值法
        for contract in Contracts:
            median = df[contract].median()
            mad = np.abs(df[contract] - median).median() # 计算绝对中位差
            
            if mad==0:
                print(f"MAD is 0 for {factor_index} contract {contract}. Skipping depolarization.")
                df_depolarized[contract] = df[contract]
            else:
                # 设置上下限，超出该范围的值将被替换
                lower_bound = median - 3 * mad
                upper_bound = median + 3 * mad
                
                # 使用上下限对数据进行限制
                df_depolarized[contract]=df[contract].clip(lower_bound, upper_bound)
            
                
        #z_score标准化因子
            if df_depolarized[contract].std() == 0:
                print(f"Standard deviation is 0 for {factor_index} contract {contract}. Skipping standardization.")
                df_standardized[contract] = df_depolarized[contract]  # 保留去极值后的数据
            else:
                df_standardized[contract]=(df_depolarized[contract]-df_depolarized[contract].mean())/df_depolarized[contract].std()

        df_standardized.to_csv(f'191alpha_Factors/{frequency}{factor_index}preprocessed.csv')
