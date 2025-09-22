import datetime
import numpy as np
import pandas as pd
import os

frequencies = ['15m_','30m_','60m_']
Contracts = ['IF00','IF01','IF02','IH00','IH01','IH02','IC00','IC01','IC02','IM00','IM01','IM02']


for freq in frequencies:
    
    Returns=pd.DataFrame()
    
    filename=freq+'close.csv'
    
    Data_close=pd.read_csv('data_raw/'+filename)
    
    Data_close['trade_time']=pd.to_datetime(Data_close['trade_time'], format='%Y-%m-%d %H:%M:%S')
    Data_close.set_index(Data_close['trade_time'],inplace=True)
    
    filename=freq+'open.csv'
    
    Data_open=pd.read_csv('data_raw/'+filename)
    
    Data_open['trade_time']=pd.to_datetime(Data_open['trade_time'], format='%Y-%m-%d %H:%M:%S')
    Data_open.set_index(Data_open['trade_time'],inplace=True)
    
    for contract in Contracts:
        
        Close=Data_close[contract]
        Open=Data_open[contract]
        
        Return=(Close-Open)/Open
        
        Returns[contract]=Return
    
    Returns.to_csv('data_returns/'+freq+'Returns.csv')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        