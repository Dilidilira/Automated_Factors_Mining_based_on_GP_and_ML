import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm 

def Alpha191_compare(factors_df,lt_start_datetime='2019-11-01 08:00:00',end_datetime='2024-11-01 15:00:00',freq=15):
    """
    输入：
    factors_df:单个因子12个合约的因子值dataframe
    lt_start_datetime:回测开始日期
    end_datetime：回测结束日期
    """
    Returns=pd.read_csv(f'/hdd/hdd1/cji/Returns/{freq}m_Returns.csv',index_col='trade_time',parse_dates=['trade_time'])
    Target_Returns=Returns[(Returns.index>=lt_start_datetime) & (Returns.index<end_datetime)]
    datetime_index = Target_Returns.index
    IC_result=pd.DataFrame(index=datetime_index,columns=['Value'])#用于保存每日的IC值
                
    for t,date_time in enumerate(datetime_index):

        Factors_data=factors_df.loc[date_time]
        if t < len(datetime_index)-1:            
            pos=Target_Returns.index.get_loc(date_time)
            Nextstep_returns=Target_Returns.iloc[pos+1]
        else:
            pos=factors_df.index.get_loc(date_time)
            Nextstep_returns=Returns.iloc[pos+1]

        correlation = Nextstep_returns.corr(Factors_data)
        IC_result.loc[date_time]['Value']=correlation
    IC_result['Value'] = pd.to_numeric(IC_result['Value'], errors='coerce')#强制转换为numeric对象
    print(f"IC_result中空值个数为：{IC_result['Value'].isna().sum()}")
    mean_ic = IC_result['Value'].mean()                           # 均值
    std_ic = IC_result['Value'].std()                             # 标准差
    ir = mean_ic / std_ic if std_ic != 0 else np.nan     # 信息比率 IR（均值 / 标准差）
    t_value, p_value = stats.ttest_1samp(IC_result['Value'].dropna(), 0)   # t 值和 p 值
    skewness = IC_result['Value'].skew()                          # 偏度
    kurtosis = IC_result['Value'].kurtosis()                      # 峰度
  
    statistics=[mean_ic,std_ic,ir,t_value,p_value,skewness,kurtosis]
    Results=pd.DataFrame([statistics],columns=['IC mean','IC std','IR','t_value','p','IC Skew','IC kurtosis'])

    return Results


alphas191_index=pd.read_csv('/hdd/hdd1/cji/alphas191_index.csv')
factor_indexes=alphas191_index['Alpha'].to_list()
CompareTable = pd.DataFrame()#储存alpha191各个因子的各类统计指标

for factor_index in tqdm(factor_indexes):
    
    df=pd.read_csv(f'/hdd/hdd1/cji/191alpha_Factors/15m_{factor_index}preprocessed.csv')
    df['trade_time']=pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df.set_index(df['trade_time'],inplace=True)

    Results=Alpha191_compare(df)
    CompareTable=pd.concat([CompareTable,Results],ignore_index=True)
    
CompareTable['Alpha_index']=factor_indexes
CompareTable.to_csv('/hdd/hdd1/cji/alphas191_stats.csv',index=False)


