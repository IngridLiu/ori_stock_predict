from torch.autograd import Variable

# split the tensor data into train, eval, test part
def split_data(data, train_ratio, eval_ratio):
    data_size = data.size(0)

    # the size of each part data
    train_size = int(data_size * train_ratio)
    eval_size = int(data_size * eval_ratio)

    # get each part data
    train_data = data[:train_size]
    eval_data = data[train_size : train_size+eval_size]
    test_data = data[train_size+eval_size : data_size]

    return train_data, eval_data, test_data

# Batchify the whole dataset
def select_data(cuda, data, bsz):
    try:
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)

        if cuda:
            data = data.cuda()
    except:
        nbatch = len(data) // bsz
        data = data[:nbatch * bsz]

    return data

# Retrieve a batch from the source
def get_batch(news_data, news_count, stock_data, labels, i, batch_size, cuda, evaluation=False):

    if cuda:
        news_data = Variable(news_data[i: i+batch_size].cuda(), volatile=evaluation)
        news_count = Variable(news_count[i: i+batch_size].cuda())
        stock_data = Variable(stock_data[i: i+batch_size].cuda())
        labels = Variable(labels[i: i+batch_size].view(-1).cuda())
    else:
        news_data = Variable(news_data[i: i+batch_size], volatile=evaluation)
        news_count = Variable(news_count[i: i+batch_size])
        stock_data = Variable(stock_data[i: i+batch_size])
        labels = Variable(labels[i: i+batch_size].view(-1))

    return news_data, news_count, stock_data, labels

# Unpack the hidden state
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    # if type(h) == Variable:
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)