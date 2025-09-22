import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
import os


if not os.path.exists('Results'):
    os.mkdir('Results')

##############################################################

lt_start_datetime='2019-11-01 08:00:00'
st_start_datetime='2023-11-01 08:00:00'
end_datetime='2024-11-01 15:00:00'

frequencies = ['15m_','30m_','60m_']
Contracts = ['IF00','IF01','IF02','IH00','IH01','IH02','IC00','IC01','IC02','IM00','IM01','IM02']
result_info = ['IC mean','IC std','IR','t_value','p','IC Skew','IC kurtosis']

factor_index='067'

##############################################################

#创建factor专属文件夹

if not os.path.exists(f'Results/factor_alpha{factor_index}'):
    os.mkdir(f'Results/factor_alpha{factor_index}')

st_start_datetime = datetime.datetime.strptime(st_start_datetime, '%Y-%m-%d %H:%M:%S')
lt_start_datetime = datetime.datetime.strptime(lt_start_datetime, '%Y-%m-%d %H:%M:%S')
end_datetime = datetime.datetime.strptime(end_datetime, '%Y-%m-%d %H:%M:%S')

Final_result=pd.DataFrame()

for term,start_datetime in enumerate([lt_start_datetime,st_start_datetime]):
    
    Results=pd.DataFrame(index=frequencies,columns=result_info)#用于保存IC的统计量
    
    for freq in frequencies:
        
        Returns=pd.read_csv(f'Returns/{freq}Returns.csv',index_col='trade_time',parse_dates=['trade_time'])
        Factors=pd.read_csv(f'191alpha_Factors/{freq}alpha{factor_index}preprocessed.csv',index_col='trade_time',parse_dates=['trade_time'])
        
        Target_Returns=Returns[(Returns.index>=start_datetime) & (Returns.index<end_datetime)]
        datetime_index = Target_Returns.index
        
        IC_result=pd.DataFrame(index=datetime_index,columns=['Value'])#用于保存每日的IC值
                
        for t,date_time in enumerate(datetime_index):
            
            Factors_data=Factors.loc[date_time]
            if t < len(datetime_index)-1:            
                pos=Target_Returns.index.get_loc(date_time)
                Nextstep_returns=Target_Returns.iloc[pos+1]
            else:
                pos=Factors.index.get_loc(date_time)
                Nextstep_returns=Returns.iloc[pos+1]
            
            correlation = Nextstep_returns.corr(Factors_data)
            IC_result.loc[date_time]['Value']=correlation
            IC_result['Value'] = pd.to_numeric(IC_result['Value'], errors='coerce')#强制转换为numeric对象
        
        # 保存IC的时间序列
        if term==0:
            IC_result.to_csv(f'Results/factor_alpha{factor_index}/{freq}Factor{factor_index}_IC_longterm_Results.csv')
        if term==1:
            IC_result.to_csv(f'Results/factor_alpha{factor_index}/{freq}Factor{factor_index}_IC_shortterm_Results.csv')
        
        # 计算统计量
        print(f"IC_result中空值个数为：{IC_result['Value'].isna().sum()}")
        mean_ic = IC_result['Value'].mean()                           # 均值
        std_ic = IC_result['Value'].std()                             # 标准差
        ir = mean_ic / std_ic if std_ic != 0 else np.nan     # 信息比率 IR（均值 / 标准差）
        t_value, p_value = stats.ttest_1samp(IC_result['Value'], 0)   # t 值和 p 值
        skewness = IC_result['Value'].skew()                          # 偏度
        kurtosis = IC_result['Value'].kurtosis()                      # 峰度

        # 将结果保存到 Results 中
        Results.loc[freq, 'IC mean'] = mean_ic
        Results.loc[freq, 'IC std'] = std_ic
        Results.loc[freq, 'IR'] = ir
        Results.loc[freq, 't_value'] = t_value
        Results.loc[freq, 'p'] = p_value
        Results.loc[freq, 'IC Skew'] = skewness
        Results.loc[freq, 'IC kurtosis'] = kurtosis
    
    Results['Time Period'] = f'{start_datetime.strftime("%Y-%m-%d %H:%M:%S")} - {end_datetime.strftime("%Y-%m-%d %H:%M:%S")}'
    Final_result = pd.concat([Final_result, Results], axis=1)

Final_result = Final_result.T