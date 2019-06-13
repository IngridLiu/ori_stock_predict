import os
import csv
import torch

import sys
sys.path.append('../')

# Data Class
class Dictionary(object):
    def __init__(self):
        self.date2idx = {}
        self.idx2date = []

    def add_date(self, date):
        if date not in self.date2idx:
            self.idx2date.append(str(date))
            self.date2idx[date] = len(self.idx2date) - 1

    def __len__(self):
        return len(self.date2idx)

class Corpus(object):
    def __init__(self, news_path, stock_path, emsize, N):
        self.dictionary = Dictionary()

        # Load all train, valid and test data
        self.news_data, self.news_counts, self.stock_data, self.labels = self.tokenize(news_path, stock_path, emsize, N)

    def tokenize(self, news_path, stock_path, emsize, N):
        '''tokenize the csv file of news and stock data '''
        assert os.path.exists(news_path)
        assert os.path.exists(stock_path)

        # Add date to the dictionary
        date_count = 0   # 数据量的大小（其实表示一共包含多少天的数据）
        daily_news_count_list = [] # 存储每天的idx对应的新闻数目的大小
        daily_count = 0
        max_daily_count = 0 # 获取每天新闻数目的最大值
        stock_len = 0   # 每天stock数据的长度

        # get date_count, max_daily_count,
        with open(stock_path, 'r') as f:
            Reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for record in Reader:
                date_count += 1
                date = record[-1]
                daily_count = int(float(record[-4]))
                daily_news_count_list.append(daily_count)
                stock_len = len(record) - 3
                if daily_count > max_daily_count:
                    max_daily_count = daily_count
                self.dictionary.add_date(date)

        # read news data
        daily_news_data_list = []
        with open(news_path, 'r') as f:
            Reader = csv.reader(f, delimiter = ',', quoting=csv.QUOTE_MINIMAL)
            for idx, record in enumerate(Reader):
                daily_news_data_list.append([float(x) for x in record[1].replace('[', '').replace(']', '').split()])


        # Tokenize stock data
        with open(stock_path, 'r') as f:
            Reader = csv.reader(f, delimiter = ',', quoting=csv.QUOTE_MINIMAL)
            daily_stock_data = torch.FloatTensor(date_count, stock_len).fill_(0)
            daily_labels = torch.FloatTensor(date_count).fill_(0)

            for date_idx, record in enumerate(Reader):
                daily_labels[date_idx] = int(float(record[-5]))
                for i in range(stock_len):
                    daily_stock_data[date_idx, i] = float(record[i])

        # Tokenize news data
        with open(stock_path, 'r') as f:
            Reader = csv.reader(f, delimiter = ',', quoting=csv.QUOTE_MINIMAL)
            daily_news_data = torch.FloatTensor(date_count, max_daily_count, emsize).fill_(0)
            for idx, record in enumerate(Reader):
                date = int(float(record[-1]))
                daily_news_count = int(float(record[-4]))
                start_index = int(float(record[-3]))
                end_index = int(float(record[-2]))

                daily_news_data[idx, :daily_news_count] = torch.FloatTensor(daily_news_data_list[start_index : end_index])

        data_size = daily_stock_data.size(0) - N
        news_data = torch.FloatTensor(data_size, N, max_daily_count, emsize)
        news_count = torch.LongTensor(data_size, N)
        stock_data = torch.FloatTensor(data_size, N, stock_len)
        labels = torch.LongTensor(data_size)
        for idx in range(N, daily_stock_data.size(0)):
            news_data[idx-N] = daily_news_data[idx-N: idx]
            news_count[idx-N] = torch.FloatTensor(daily_news_count_list[idx-N: idx])
            stock_data[idx-N] = daily_stock_data[idx-N: idx]
            labels[idx-N] = daily_labels[idx]

        return news_data, news_count, stock_data, labels



      

