#!/usr/bin/python
# -*- coding UTF-8 -*-
import os
import pandas as pd
import numpy as np

import sys
sys.path.append('../')
from cfg import *


# 加载数据
stock_data = pd.read_csv(data_root + stock_file_name, header = 0, parse_dates=['trade_date'])
# print(stock_data.index)

# 对行业板块进行热独编码
# onehot_ts_code = pd.get_dummies(stock_data['ts_code']).astype('float')
# stock_data = pd.merge(onehot_ts_code, stock_data, left_index=True, right_index=True)


# 按行业板块分组
ts_group = stock_data.groupby('ts_code')
# 按行业板块对数据分别进行处理

trade_keys = []
for key, ts_df in ts_group:

    trade_keys.append(key)

    # 把trade_date列设置为索引列:
    ts_df.set_index('trade_date', inplace=True)

    # 补充缺失日期
    dt = pd.date_range('20140414', '20190401')
    idx = pd.DatetimeIndex(dt)
    ts_df = ts_df.reindex(idx)

    # 补充缺失数据
    # 参考上一行的值填充;
    ts_df = ts_df.fillna(method='ffill')

    # 设定predict目标值
    ts_df['predict'] = ts_df['y'].shift(-1)

    # 添加每天的新闻索引三列,news_start_index, news_end_index, news_count
    saved_news_df = pd.read_csv(os.path.join(data_root, preprocess_news_file), header=0)
    daily_news_count_series = saved_news_df['date'].value_counts()
    daily_news_info = []
    start_index = 0
    end_index = 0
    for date in dt:
        str_date = date.strftime('%Y%m%d')
        daily_news_count = daily_news_count_series[int(str_date)]
        end_index = start_index + daily_news_count
        daily_news_info.append([str_date, daily_news_count, start_index, end_index])
        start_index = end_index
    daily_news_count_df = pd.DataFrame(daily_news_info, columns=['date', 'news_count', 'start_index', 'end_index'])
    daily_news_count_df.set_index('date', inplace=True)
    daily_news_count_df.reindex(idx)
    ts_df = ts_df.join(daily_news_count_df, how = 'left')

    # 删除最后一行的数据
    ts_df = ts_df.drop(ts_df.index[len(ts_df)-1])

    # 数值数据标准化、归一化
    num_columns = ['open', 'low', 'high', 'close', 'change', 'pct_change', 'vol', 'amount', 'pe', 'pb']
    for num_column in num_columns:
        # 标准化
        ts_df[num_column] = (ts_df[num_column] - ts_df[num_column].min())/(ts_df[num_column].max()-ts_df[num_column].min())
        # 归一化
        ts_df[num_column] = ts_df[num_column] / np.sqrt((np.power(ts_df[num_column], 2)).sum())


    ts_df['date'] = ts_df.index
    ts_df['date'] = ts_df['date'].apply(lambda x: x.strftime('%Y%m%d'))


    print(ts_df.columns)
    print(ts_df.head(10))

    # 删除y列
    ts_df.drop(['y', 'ts_code', 'name'], axis=1, inplace=True)
    print(ts_df.head(10))
    print(ts_df.columns)

    # 按行业分别 save processed stock data
    stock_save_name = 'processed_TRAINSET_STOCK/' + str(key) + '.csv'
    ts_df.to_csv(data_root + stock_save_name, index=False, header=True)

    # save stock data for input
    saved_stock_file = 'input_TRAINSET_STOCK/' + str(key) + '.csv'
    ts_df.to_csv(data_root + saved_stock_file, index=False, header=False)

pd.DataFrame(trade_keys).to_csv(data_root + 'trade_keys.csv', header=False, index=False)
print("Success to handle stock data!")