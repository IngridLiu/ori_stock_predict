import jieba
import numpy as np
import pandas as pd

# # 加载数据
# file_path = '/home/ingrid/Data/stockpredict/'
# news_file_name = 'processed_TRAINSET_NEWS.csv'
#
# news_data = pd.read_csv(file_path + news_file_name, header=0)
# print(news_data.columns)

# str_1 = '文件路径，如果没有指定则将会直接返回字符串的json'
# print(' '.join(jieba.cut(str_1)))

data = [[1, 1, 1],
        [1, 1, 2],
        [1, 2, 1],
        [1, 2, 2],
        [2, 1, 1],
        [2, 1, 2]]
columns =['user_id', 'action_type', 'action_time']
data_df = pd.DataFrame(data, columns = columns)
data_df.groupby('user_id')['action_time'].apply(lambda i:i.iloc[-1])

print(data_df)