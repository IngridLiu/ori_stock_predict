#!/usr/bin/python
# -*- coding UTF-8 -*-
import re
import jieba
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# 加载数据
file_path = '/home/ingrid/Data/stockpredict/'
news_file_name = 'TRAINSET_NEWS.csv'

news_data = pd.read_csv(file_path + news_file_name, header=0)



# 去除数据中的非文本部分
# 过滤不了\\ \ 中文（）还有————
r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'#用户也可以在此进行自定义过滤字符
# 者中规则也过滤不完全
r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
# \\\可以过滤掉反向单杠和双杠，/可以过滤掉正向单杠和双杠，第一个中括号里放的是英文符号，第二个中括号里放的是中文符号，第二个中括号前不能少|，否则过滤不完全
r3 =  "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
# 去掉括号和括号内的所有内容
r4 =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
# 去掉html
r5 = '<.*?>'
news_data['title'] = news_data['title'].apply(lambda sen: re.sub(r1, '', str(sen)))
news_data['content'] = news_data['content'].apply(lambda sen:re.sub(r1, '', str(sen)))


# 分词
news_data['sen_seg'] = news_data['content'].apply(lambda sen:' '.join(jieba.cut(sen)))


# 去除停用词
def drop_stopwords(sen_seg, stopwords):
    sen_clean = []
    for word in sen_seg.split():
        if word in stopwords:
            continue
        sen_clean.append(word)
    return ' '.join(sen_clean)

stopwords_file_name = 'hagongda_stopwords.txt'
stopwords_file_path = file_path + stopwords_file_name
stopwords_file = open(stopwords_file_path)
stopwords = stopwords_file.readlines()
news_data['clean_seg'] = news_data['sen_seg'].apply(lambda sen_seg:drop_stopwords(sen_seg, stopwords))


# 去除常见词
freq_words = pd.Series(str(news_data['clean_seg']).split()).value_counts()[:10]
freq_words = list(freq_words.index)
news_data['clean_seg'] = news_data['clean_seg'].apply(lambda sen:' '.join(x for x in sen.split() if x not in freq_words))


# 稀缺词去除
scarce_words = pd.Series(str(news_data['clean_seg']).split()).value_counts()[-10:]
scarce_words = list(scarce_words.index)
news_data['clean_seg'] = news_data['clean_seg'].apply(lambda sen: ' '.join(x for x in sen.split() if x not in scarce_words))


# word2vec词向量表示文本
def sen_vec(sen_seg, train_w2c):
    vec = np.zeros(500)
    count = 0
    for word in sen_seg.split():
        try:
            vec += train_w2v[word]
            count += 1
        except:
            pass
    return vec/count

train_w2v = Word2Vec(news_data['clean_seg'], min_count = 5, size = 500, workers = 4)   # 词频少于min_count次数的单词会被丢弃掉, size指特征向量的维度为50, workers参数控制训练的并行数;
news_data['word2vec'] = news_data['clean_seg'].apply(lambda sen_seg: sen_vec(sen_seg, train_w2v))

# 数据存储
save_news_name = 'processed_TRAINSET_NEWS.csv'
news_data.to_csv(file_path + save_news_name, sep = ',', header = True, index = False)


