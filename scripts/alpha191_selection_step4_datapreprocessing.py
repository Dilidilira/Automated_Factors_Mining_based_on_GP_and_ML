import datetime
import numpy as np
import pandas as pd

def merge_alpha_dataframes(alpha_dict):
    # 将所有 alpha DataFrame 的索引和列对齐
    for key, df in alpha_dict.items():
        alpha_dict[key] = df.stack().to_frame(name=key)  # stack 转换列为 MultiIndex

    # 合并所有 stacked DataFrame
    merged_df = pd.concat(alpha_dict.values(), axis=1)
    return merged_df.reset_index()

def filter_by_time_ranges(df, time_ranges):
    condition = False
    for start, end in time_ranges:
        condition |= (df['trade_time'] >= start) & (df['trade_time'] <= end)
    return df[condition]

def get_next_time_returns(df, returns):
    """
    为每个时间点的行找到对应合约的下一个时间点收益率。
    """
    df=df.copy()
    df['next_time'] = df['trade_time'].apply(lambda t: returns.index[returns.index.get_loc(t) + 1] if t in returns.index[:-1] else np.nan)# 找到下一个时间点
    df['Y'] = df.apply(lambda row: returns.loc[row['next_time'], row['level_1']] if row['next_time'] in returns.index else np.nan, axis=1)
    return df

if __name__ == "__main__":

    freq = '15m_'#'15m_','30m_','60m_'
    alphas191_index=pd.read_csv('/hdd/hdd1/cji/Trees/Non_redundant_factors.csv')
    factor_indexes=alphas191_index['Factor_Index'].to_list()


    #########预处理###################################

#    alpha_dict={}
#    for factor_index in factor_indexes:
#        Factors_data=pd.read_csv(f"191alpha_Factors/{freq}{factor_index}preprocessed.csv")
#        Factors_data['trade_time']=pd.to_datetime(Factors_data['trade_time'], format='%Y-%m-%d %H:%M:%S')
#        Factors_data=Factors_data.set_index('trade_time',drop=True)
#        alpha_dict[factor_index]=Factors_data

#    merged_df = merge_alpha_dataframes(alpha_dict)

    merged_df=pd.read_csv('/hdd/hdd1/cji/Trees/alphas191_data.csv')
    # 设置训练集的时间范围
    train_time_ranges = [
        ('2020-01-22 08:00:00', '2022-01-22 15:00:00'),
        ('2023-01-01 08:00:00', '2023-12-31 15:00:00')
    ]
    # 设置测试集的时间范围
    test_time_ranges = [
        ('2022-01-23 08:00:00', '2022-07-22 15:00:00'),
        ('2024-01-01 08:00:00', '2024-03-01 15:00:00')
    ]

    # 设置X_train, X_test
    filtered_df_train = filter_by_time_ranges(merged_df, train_time_ranges)
    filtered_df_test = filter_by_time_ranges(merged_df, test_time_ranges)

    X_train = filtered_df_train.loc[:, factor_indexes]
    X_test = filtered_df_test.loc[:, factor_indexes]

    # 设置y_train, y_test
#    returns=pd.read_csv(f'Returns/{freq}Returns.csv')
#    returns['trade_time']=pd.to_datetime(returns['trade_time'], format='%Y-%m-%d %H:%M:%S')
#    returns=returns.set_index('trade_time',drop=True)

#    labeled_df_train = get_next_time_returns(filtered_df_train, returns)
#    Y_train = labeled_df_train['Y']
#    labeled_df_test = get_next_time_returns(filtered_df_test, returns)
#    Y_test = labeled_df_test['Y']
    
    Y_train = filtered_df_train['Y']
    Y_test = filtered_df_test['Y']

    #保存训练集测试集
    X_train.to_csv('/hdd/hdd1/cji/Trees/X_train_selected.csv')
    X_test.to_csv('/hdd/hdd1/cji/Trees/X_test_selected.csv')
    Y_train.to_csv('/hdd/hdd1/cji/Trees/Y_train_selected.csv')
    Y_test.to_csv('/hdd/hdd1/cji/Trees/Y_test_selected.csv')
    ###########################################################################################

