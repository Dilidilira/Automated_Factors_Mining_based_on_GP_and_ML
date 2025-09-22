import datetime
import numpy as np
import pandas as pd
import os
from alphas191 import *
import tqdm 

def main():
    frequencies = ['15m_']
    Contracts = ['IF00','IF01','IF02','IH00','IH01','IH02','IC00','IC01','IC02','IM00','IM01','IM02']
    infos = ['amount','close','high','low','open','open_interest','volume']

    for freq in tqdm.tqdm(frequencies):
        
        d_list=[]
        
        for contract in Contracts:
            
            d=pd.DataFrame()#合约（如IF00）的面板数据
            
            for info in infos:
                
                filename='data_raw/'+freq+info+'.csv'           
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
        panel_data_after=panel_data[panel_data.index>'2022-7-21 15:00:00']

        instance = Alphas191(panel_data_before)
        Factors_before_dic={}
        Factors_after_dic={}
        attr_name_list=[]

        for attr_name in tqdm.tqdm(dir(instance),leave=False):
            if attr_name.startswith("alpha") and callable(getattr(instance, attr_name)):
                method = getattr(instance, attr_name)
                Factors_before = method()
                #inf值处理
                Factors_before=Factors_before.replace([np.inf, -np.inf], np.nan)
                #空值处理
                columns_to_fill = ['IF00','IF01','IF02','IH00','IH01','IH02','IC00','IC01','IC02','IM00','IM01','IM02'] 
                for col in columns_to_fill:
                    if col in Factors_before.columns:
                        Factors_before[col] = Factors_before[col].fillna(Factors_before[col].mean())
                #处理后的Factors保存到字典中
                Factors_before_dic[attr_name]=Factors_before
                attr_name_list.append(attr_name)

        instance = Alphas191(panel_data_after)

        for attr_name in tqdm.tqdm(dir(instance),leave=False):
            if attr_name.startswith("alpha") and callable(getattr(instance, attr_name)):
                method = getattr(instance, attr_name)
                Factors_after = method()
                #inf值处理
                Factors_after=Factors_after.replace([np.inf, -np.inf], np.nan)
                #空值处理
                columns_to_fill = ['IF00','IF01','IF02','IH00','IH01','IH02','IC00','IC01','IC02','IM00','IM01','IM02'] 
                for col in columns_to_fill:
                    if col in Factors_after.columns:
                        Factors_after[col] = Factors_after[col].fillna(Factors_after[col].mean())
                Factors_after_dic[attr_name]=Factors_after
        
        for attr_name in attr_name_list:
            Factors=pd.concat([Factors_before_dic[attr_name],Factors_after_dic[attr_name]])    
            Factors.to_csv('factors/alpha191/'+freq+f"{attr_name}.csv")

    alphas191_index = pd.DataFrame(attr_name_list, columns=['Alpha'])
    alphas191_index.to_csv('factors/alpha191/alphas191_index.csv')

if __name__ == "__main__":
    main()