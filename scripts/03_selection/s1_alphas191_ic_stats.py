import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm 

def Alpha191_compare(factors_df,start_datetime='2019-11-01 08:00:00',end_datetime='2024-11-01 15:00:00',freq=15):
    """
    Inputs
    ----------
    factors_df          : 单个因子12个合约的因子值df
    start_datetime      : 回测开始日期
    end_datetime        : 回测结束日期
    
    """
    Returns=pd.read_csv(f'data_returns/{freq}m_Returns.csv',index_col='trade_time',parse_dates=['trade_time'])
    Target_Returns=Returns[(Returns.index>=start_datetime) & (Returns.index<end_datetime)]
    datetime_index = Target_Returns.index
    IC_result=pd.DataFrame(index=datetime_index,columns=['Value'])
                
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
    IC_result['Value'] = pd.to_numeric(IC_result['Value'], errors='coerce')
    print(f"IC_result中空值个数为：{IC_result['Value'].isna().sum()}")
    # ic mean
    mean_ic = IC_result['Value'].mean()      
    # ic std                     
    std_ic = IC_result['Value'].std()
    # IR (ic mean/ic std)                             
    ir = mean_ic / std_ic if std_ic != 0 else np.nan     
    # t-value and p-value
    t_value, p_value = stats.ttest_1samp(IC_result['Value'].dropna(), 0)   
    # skewness
    skewness = IC_result['Value'].skew()                          
    # kurtosis
    kurtosis = IC_result['Value'].kurtosis()
  
    statistics=[mean_ic,std_ic,ir,t_value,p_value,skewness,kurtosis]
    Results=pd.DataFrame([statistics],columns=['IC mean','IC std','IR','t_value','p','IC Skew','IC kurtosis'])

    return Results

def main():
    
    alphas191_index=pd.read_csv('factors/alpha191/alphas191_index.csv')
    factor_indexes=alphas191_index['Alpha'].to_list()
    CompareTable = pd.DataFrame()

    for factor_index in tqdm(factor_indexes):
        
        df=pd.read_csv(f'factors/alpha191/15m_{factor_index}_preprocessed.csv')
        df['trade_time']=pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
        df.set_index(df['trade_time'],inplace=True)

        Results=Alpha191_compare(df)
        CompareTable=pd.concat([CompareTable,Results],ignore_index=True)
        
    CompareTable['Alpha_index']=factor_indexes
    CompareTable.to_csv('results/alphas191_ic_stats.csv',index=False)


if __name__ == "__main__":
    main()