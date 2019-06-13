import argparse
import time
import torch.nn as nn

from cfg import *
from data_preprocess.data_input import *
from Utility import *
from model import *

##########################################################################
#                         CommandLine Argument Setup                     #
##########################################################################

parser = argparse.ArgumentParser(description='PyTorch Self-Attentive Sentence Embedding Model')
parser.add_argument('-f', default='self', help='To make it runnable in jupyter' )
parser.add_argument('--data_root', type=str, default=data_root, help='location of the data corpus')
parser.add_argument('--news_file', type=str, default=input_news_file, help='the file name of news data.')
parser.add_argument('--stock_dir', type=str, default=input_stock_path, help='the file name of stock data.')
parser.add_argument('--default_trade', type = str, default='801010', help='the trade of stock to predict')

parser.add_argument('--train_ratio', type = float, default='0.6', help='the ratio of train set of whole data.')
parser.add_argument('--eval_ratio', type = float, default='0.2', help='the ratio of eval set of whole data.')

parser.add_argument('--model', type=str, default='basic_model', help='the model to train')
#parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
parser.add_argument('--N', type=int, default=20, help='the size of input days')
parser.add_argument('--nhid', type=int, default=300, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
# parser.add_argument('--r', type=int, default=30, help='r in paper, # of keywords you want to focus on')
# parser.add_argument('--mlp_nhid', type=int, default=300, help='r in paper, # of keywords you want to focus on')
# parser.add_argument('--da', type=int, default=350, help='da in paper' )
# parser.add_argument('--lamb', type=float, default=1, help='penalization term coefficient')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='sgd with momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N', help='batch size')
parser.add_argument('--eval_size', type=int, default=32, metavar='N', help='evaluation batch size')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--pretrained', type=str, default='', help='whether start from pretrained model')
parser.add_argument('--cuda', default=False, action='store_true', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='report interval')
parser.add_argument('--save', type=str,  default='stock_predict.pt', help='path to save the final model')

args = parser.parse_args()

news_path = os.path.join(args.data_root, args.news_file)
stock_path = os.path.join(args.data_root, args.stock_dir, args.default_trade + '.csv')
corpus = Corpus(news_path, stock_path, args.emsize, args.N)
print('Training data loaded...')

# split the dataset
train_news_data, eval_news_data, test_news_data = split_data(corpus.news_data, args.train_ratio, args.eval_ratio)
train_news_count, eval_news_count, test_news_count = split_data(corpus.news_counts, args.train_ratio, args.eval_ratio)
train_stock_data, eval_stock_data, test_stock_data = split_data(corpus.stock_data, args.train_ratio, args.eval_ratio)
train_labels, eval_labels, test_labels = split_data(corpus.labels, args.train_ratio, args.eval_ratio)
print('Splited data into train, test, eval part...')

# Make Dataset batchifiable
train_news_data = select_data(args.cuda, train_news_data, args.batch_size)
train_news_count = select_data(args.cuda, train_news_count, args.batch_size)
train_stock_data = select_data(args.cuda, train_stock_data, args.batch_size)
train_labels = select_data(args.cuda, train_labels, args.batch_size)

eval_news_data = select_data(args.cuda, eval_news_data, args.batch_size)
eval_news_count = select_data(args.cuda, eval_news_count, args.batch_size)
eval_stock_data = select_data(args.cuda, eval_stock_data, args.batch_size)
eval_labels = select_data(args.cuda, eval_labels, args.batch_size)

test_news_data = select_data(args.cuda, test_news_data, args.batch_size)
test_news_count = select_data(args.cuda, test_news_count, args.batch_size)
test_stock_data = select_data(args.cuda, test_stock_data, args.batch_size)
test_labels = select_data(args.cuda, test_labels, args.batch_size)
print('Make data batchfiable...')


# define model
ntokens = len(corpus.dictionary)
nclass = 2

if not args.pretrained:
    model = BasicModel()
else:
    model = torch.load(args.pretrained)
    print('Pretrained model loaded.')


entropy_loss = nn.CrossEntropyLoss()
if args.cuda:
    model.cuda()
    entropy_loss.cuda()

# Define the training function
def train(lr, epoch):
    # word_update: whether glove vectors are updated
    total_loss = 0
    start_time = time.time()
    all_losses = []

    news_hidden = model.init_hidden(args.batch_size)

    # per-parameter training
    params = list(model.parameters())

    optimizer = torch.optim.SGD(params, lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for batch_idx, start_idx in enumerate(range(0, train_news_data.size(0)-1, args.batch_size)):

        # Retrieve one batch for training
        news_data, news_count, stock_date, labels = get_batch(train_news_data, train_news_count, train_stock_data, train_labels, start_idx, args.N, args.batch_size, args.cuda)
        hidden = repackage_hidden(news_hidden)
























