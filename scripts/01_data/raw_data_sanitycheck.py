import numpy as np
import pandas as pd
from datetime import time

infos = ['amount','close','high','low','open','open_interest','volume']

# —— 15m vs 30m：找“日期差异” ——
for info in infos:
    df_30 = pd.read_csv(f'data_raw/30m_{info}.csv')
    df_30['trade_time'] = pd.to_datetime(df_30['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df_30.set_index('trade_time', inplace=True)
    df_30.index = pd.DatetimeIndex(df_30.index)           # 关键：确保是 DatetimeIndex

    df_15 = pd.read_csv(f'data_raw/15m_{info}.csv')
    df_15['trade_time'] = pd.to_datetime(df_15['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df_15.set_index('trade_time', inplace=True)
    df_15.index = pd.DatetimeIndex(df_15.index)           # 关键：确保是 DatetimeIndex

    # 用 normalize() 仅比较“日期”
    df_15_only = df_15[~df_15.index.normalize().isin(df_30.index.normalize())]
    print(f"15m_{info} 中不在 30m 的日期条数：{len(df_15_only)}")

# —— 60m：统计各固定时刻的条数 ——
for info in infos:
    df = pd.read_csv(f'data_raw/60m_{info}.csv')
    df['trade_time'] = pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('trade_time', inplace=True)
    df.index = pd.DatetimeIndex(df.index)

    # 用 indexer_at_time() 或 index.time 计数
    print(f"60m_{info} 10:30 条数：{len(df.index.indexer_at_time('10:30'))}")
    print(f"60m_{info} 11:30 条数：{len(df.index.indexer_at_time('11:30'))}")
    print(f"60m_{info} 14:00 条数：{len(df.index.indexer_at_time('14:00'))}")
    print(f"60m_{info} 15:00 条数：{len(df.index.indexer_at_time('15:00'))}")

# —— 15m：统计一组时刻 ——
times_15 = ['09:45','10:00','10:15','10:30','10:45','11:00','11:15','11:30',
            '13:15','13:30','13:45','14:00','14:15','14:30','14:45','15:00']
for info in infos:
    df = pd.read_csv(f'data_raw/15m_{info}.csv')
    df['trade_time'] = pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('trade_time', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    for t in times_15:
        print(f"15m_{info} {t} 条数：{len(df.index.indexer_at_time(t))}")

# —— 30m：统计一组时刻 ——
times_30 = ['10:00','10:30','11:00','11:30','13:30','14:00','14:30','15:00']
for info in infos:
    df = pd.read_csv(f'data_raw/30m_{info}.csv')
    df['trade_time'] = pd.to_datetime(df['trade_time'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('trade_time', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    for t in times_30:
        print(f"30m_{info} {t} 条数：{len(df.index.indexer_at_time(t))}")