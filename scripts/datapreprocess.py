import datetime
import numpy as np
import pandas as pd
import os

infos = ['amount','close','high','low','open','open_interest','volume']

for info in infos:

    df_30 = pd.read_csv('30m_'+info+'.csv')
    df_30['trade_time']=pd.to_datetime(df_30['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df_30.set_index(df_30['trade_time'],inplace=True)

    df_15 = pd.read_csv('15m_'+info+'.csv')
    df_15['trade_time']=pd.to_datetime(df_15['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df_15.set_index(df_15['trade_time'],inplace=True)

    df_15 = df_15[~df_15.index.strftime('%Y-%m-%d').isin(df_30.index.strftime('%Y-%m-%d'))]
    print(f"df_15的长度为：{len(df_15)}")

for info in infos:

    df = pd.read_csv('60m_'+info+'.csv')
    df['trade_time']=pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df.set_index(df['trade_time'],inplace=True)

    df_1 = df[~df.index.strftime('%H:%M:%S').isin(['10:00:00', '11:00:00'])]

    print(f"60m_{info}的10:30的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['10:30:00'])])}")
    print(f"60m_{info}的11:30的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['11:30:00'])])}")
    print(f"60m_{info}的14:00的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['14:00:00'])])}")
    print(f"60m_{info}的15:00的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['15:00:00'])])}")


for info in infos:

    df = pd.read_csv('15m_'+info+'.csv')
    df['trade_time']=pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df.set_index(df['trade_time'],inplace=True)

    print(f"15m_{info}的09:45的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['09:45:00'])])}")
    print(f"15m_{info}的10:00的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['10:00:00'])])}")
    print(f"15m_{info}的10:15的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['10:15:00'])])}")
    print(f"15m_{info}的10:30的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['10:30:00'])])}")
    print(f"15m_{info}的10:45的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['10:45:00'])])}")
    print(f"15m_{info}的11:00的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['11:00:00'])])}")
    print(f"15m_{info}的11:15的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['11:15:00'])])}")
    print(f"15m_{info}的11:30的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['11:30:00'])])}")
    print(f"15m_{info}的13:15的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['13:15:00'])])}")
    print(f"15m_{info}的13:30的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['13:30:00'])])}")
    print(f"15m_{info}的13:45的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['13:45:00'])])}")
    print(f"15m_{info}的14:00的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['14:00:00'])])}")
    print(f"15m_{info}的14:15的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['14:15:00'])])}")
    print(f"15m_{info}的14:30的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['14:30:00'])])}")
    print(f"15m_{info}的14:45的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['14:45:00'])])}")
    print(f"15m_{info}的15:00的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['15:00:00'])])}")

for info in infos:

    df = pd.read_csv('30m_'+info+'.csv')
    df['trade_time']=pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df.set_index(df['trade_time'],inplace=True)
    
    m30_time = ['10:00:00','10:30:00','11:00:00','11:30:00','13:30:00','14:00:00','14:30:00','15:00:00']

    print(f"30m_{info}的10:00的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['10:00:00'])])}")
    print(f"30m_{info}的10:30的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['10:30:00'])])}")
    print(f"30m_{info}的11:00的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['11:00:00'])])}")
    print(f"30m_{info}的11:30的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['11:30:00'])])}")
    print(f"30m_{info}的13:30的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['13:30:00'])])}")
    print(f"30m_{info}的14:00的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['14:00:00'])])}")
    print(f"30m_{info}的14:30的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['14:30:00'])])}")
    print(f"30m_{info}的15:00的个数为：{len(df[df.index.strftime('%H:%M:%S').isin(['15:00:00'])])}")







