import argparse
import time
import numpy as np
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
parser.add_argument('--nclass', type=int, default=2, help='the number of class to predict.')
parser.add_argument('--emb_size', type=int, default=200, help='size of word embeddings')
parser.add_argument('--N', type=int, default=20, help='the size of input days')
parser.add_argument('--nhid', type=int, default=300, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--na', type=int, default=200, help='the length of news attention vector.' )
parser.add_argument('--da', type=int, default=350, help='da in paper' )
parser.add_argument('--head', type=float, default=1, help='the number of head in model.')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--lamb', type=float, default=1, help='penalization term coefficient')
parser.add_argument('--momentum', type=float, default=0.9, help='sgd with momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N', help='batch size')
parser.add_argument('--eval_size', type=int, default=32, metavar='N', help='evaluation batch size')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--pretrained', type=str, default='', help='whether start from pretrained model')
parser.add_argument('--cuda', default=True, action='store_true', help='use CUDA')
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='report interval')
parser.add_argument('--save', type=str,  default='../model/stock_predict.pt', help='path to save the final model')

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
nclass = 2

if not args.pretrained:
    model = BasicModel(args.emb_size, args.na, args.da, args.head, args.head, args.nhid, args.nlayers, args.mlp_hid, args.nclass, args.cuda)
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

        output, hidden, news_weights, daily_weights = model(news_data, news_count, args.N, hidden)
        loss = entropy_loss(output.view(-1, nclass), labels) + args.lamb * (torch.norm(news_weights, 2) + torch.norm(daily_weights, 2))

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping in case of explosion
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        optimizer.step()

        total_loss += loss.data
        all_losses.append(loss.data)

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} |'.format(
                epoch, batch_idx, len(news_data) // args.batch_size, lr,
                                  elapsed * 1000 / args.log_interval, cur_loss))

            total_loss = 0
            start_time = time.time()

        return np.mean(all_losses)

def evaluate(news_data, len_li, labels):
    total_loss = 0

    acc = []
    pre = []
    rec = []
    f1 = []

    hidden = model.init_hidden( args.eval_size )

    for start_idx in range( 0, news_data.size(0) - 1, args.eval_size ):
        news_data, news_count, stock_date, labels = get_batch(train_news_data, train_news_count, train_stock_data,
                                                              train_labels, start_idx, args.N, args.batch_size,
                                                              args.cuda)
        output, hidden, news_weights, daily_weights = model(news_data, news_count, args.N, hidden)
        output_flat = output.view(-1, nclass)

        total_loss += news_data.size(0) * (entropy_loss(output_flat, labels).data + args.lamb * (torch.norm(news_weights, 2) + torch.norm(daily_weights, 2)))
        hidden = repackage_hidden(hidden)

        _, pred = output_flat.topk(1, 1, True, True)
        pred = pred.t()
        target = labels.view(1, -1)

        p, r, f, a = compute_measure(pred, target)
        acc.append( a )
        pre.append( p )
        rec.append( r )
        f1.append( f )

    # Compute Precision, Recall, F1, and Accuracy
    print('Measure on this dataset')
    print('Precision:', np.mean(pre))
    print('Recall:', np.mean(rec))
    print('F1:', np.mean(f1))
    print('Acc:', np.mean(acc))

    return total_loss[0] / len(news_data.size(0))


# Training Process
print('Start training...')
lr = args.lr
best_val_loss = None
all_losses = []
print('# of Epochs:', args.epochs)

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    all_losses.append(train(lr, epoch)[0])
    val_loss = evaluate(eval_news_data, eval_news_count, eval_labels)
    print('-'*80)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
          .format(epoch, (time.time() - epoch_start_time), val_loss))
    print('-'*80)

    # Save the best model and Anneal the learning rate.
    if not best_val_loss or val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(args.save, 'wb') as f:
            torch.save(model, f)
    else:
        lr /= 4.0






















