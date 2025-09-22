import datetime
import numpy as np
import pandas as pd
import os
from alphas191 import *

##################################################################################################
def factor_1(d):#######因子1：成交量增长率##########
    d['factor']=d['volume'].pct_change()
    return d

def factor_2(d):########因子2：换手率与持仓因子变化############
    d['turnover']=d['volume']/d['open_interest']
    d['open_interest_delta']=d['open_interest'].diff()
    d['factor']=d['turnover']/d['open_interest_delta'].abs()
    return d

def factor_3(d):#########因子3：换手率因子####################
    d['factor']=d['volume']/d['open_interest']
    return d

def factor_4(d):#########因子4：价格动量因子######################
    d['factor']=(d['close']-d['open'])/d['open']
    return d

#################################################################################################
if not os.path.exists('Factors'):
    os.mkdir('Factors')

frequencies = ['15m_','30m_','60m_']#'15m_','30m_','60m_'
Contracts = ['IF00','IF01','IF02','IH00','IH01','IH02','IC00','IC01','IC02','IM00','IM01','IM02']
infos = ['amount','close','high','low','open','open_interest','volume']


for freq in frequencies:
    
    d_list=[]
    
    for contract in Contracts:
        
        d=pd.DataFrame()#合约（如IF00）的面板数据
        
        for info in infos:
            
            filename=freq+info+'.csv'           
            Data=pd.read_csv(filename)     
            Data['trade_time']=pd.to_datetime(Data['trade_time'], format='%Y-%m-%d %H:%M:%S')
            Data.set_index(Data['trade_time'],inplace=True)
            
            d[info]=Data[contract]

        #计算VWAP成交量加权平均价格

        d['vwap'] = (d['close'] * d['volume']).rolling(window=2).sum() / d['volume'].rolling(window=2).sum()
        d['asset']=contract

        d_list.append(d)
    
    d_all=pd.concat(d_list)
    panel_data = d_all.pivot(columns='asset', values=['amount','close','high','low','open','open_interest','volume','vwap'])
    
    columns_to_select = pd.IndexSlice[['amount','close','high','low','open','open_interest','volume','vwap'], ['IF00','IF01','IF02','IH00','IH01','IH02','IC00','IC01','IC02']]
    panel_data_before=panel_data.loc[panel_data.index <= '2022-7-21 15:00:00', columns_to_select]
    Factors_before=Alphas191(panel_data_before).alpha105()

    panel_data_after=panel_data[panel_data.index>'2022-7-21 15:00:00']
    Factors_after=Alphas191(panel_data_after).alpha105()

    Factors_before=Factors_before.replace([np.inf, -np.inf], np.nan)
    Factors_after=Factors_after.replace([np.inf, -np.inf], np.nan)

    columns_to_fill = ['IF00','IF01','IF02','IH00','IH01','IH02','IC00','IC01','IC02','IM00','IM01','IM02'] 
    for col in columns_to_fill:
        if col in Factors_before.columns:
            Factors_before[col] = Factors_before[col].fillna(Factors_before[col].mean())
    
    for col in columns_to_fill:
        if col in Factors_after.columns:
            Factors_after[col] = Factors_after[col].fillna(Factors_after[col].mean())

    Factors=pd.concat([Factors_before,Factors_after])
    
    if not os.path.exists('191alpha_Factors'):
      os.mkdir('191alpha_Factors')

    Factors.to_csv('Factors/'+freq+'Factor_105.csv')













