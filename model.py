import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable



# the basic model, 基线模型, head = 1时
# the self-attention model, 在基线模型中引入self-attention机制, head > 1时

class BasicModel(nn.Module):
    def __init__(self, emb_size, na, da, head, nhid, nlayers, mlp_nhid, nclass, cuda):
        ''''''

        super(BasicModel, self).__init__()

        # daily news attention layer
        self.News_att_1 = nn.Linear(emb_size, na, bias = False)
        self.News_att_2 = nn.Linear(da, head, bias = False)

        # RNN layer
        self.rnn = nn.GRU( head * emb_size, nhid, nlayers, bias=True, batch_first=True, bidirectional=True)

        # daily attention layer
        self.Daily_att_1 = nn.Linear(nhid * 2, da, bias=False)
        self.Daily_att_2 = nn.Linear(da, head, bias = False)

        # final mlp layer
        self.MLP = nn.Linear(head * nhid *2, mlp_nhid)
        self.decoder = nn.Linear(mlp_nhid, nclass)

        self.init_weight()

        self.head = head
        self.nhid = nhid
        self.emb_size = emb_size
        self.nlayers = nlayers

        if cuda:
            self.cuda()

    def init_weights(self):
        init_range = 0.1
        self.News_att_1.weight.data.uniform_(-init_range, init_range)
        self.News_att_2.weight.data.uniform_(-init_range, init_range)

        self.Daily_att_1.weight.data.uniform_(-init_range, init_range)
        self.Daily_att_2.weight.data.uniform_(-init_range, init_range)

        self.MLP.weight.data.uniform_(-init_range, init_range)
        self.MLP.bias.data.fill_(0)

        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        return(Variable(weight.new(self.nlayers * 2, batch_size, self.nhid).zeros()),
               Variable(weight.new(self.nlayers * 2, batch_size, self.nhid).zeros()))

    def forward(self, input, len_li, N, hidden):
        if self.cuda():
            news_BM = Variable(torch.zeros(input.size(0), N, self.head * self.emb_size).cuda())
            daily_BM = Variable(torch.zeros(input.size(0), N, self.head * self.nhid * 2).cuda())
        else:
            news_BM = Variable(torch.zeros(input.size(0), self.head * self.emb_size))
            daily_BM = Variable(torch.zeros(input.size(0), self.head * self.nhid * 2))

        # News Attention Block
        news_weights = {}
        for i in range(input.size(0)):
            for day in range(N):
                daily_news = input[i, day, :len_li[i][day], :]
                news_att_1 = self.News_att_1(daily_news)
                news_att_2 = self.News_att_2(functional.tanh(news_att_1))

                # Attention Weights and Embedding
                news_A = functional.softmax(news_att_2.t())
                news_M = torch.mm(news_A, daily_news)
                news_BM[i, day, :] = news_M.view(-1)
                news_weights[i] = news_A

        # RNN block
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(news_BM, list(len_li.data), batch_first = True)
        output, hidden = self.rnn(rnn_input, hidden)
        depacked_output, lens = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Daily Attention Block
        daily_weights = {}
        for i in range(input.size(0)):
            daily = depacked_output[i, lens[i], :]
            daily_att_1 = self.Daily_att_1(daily)
            daily_att_2 = self.Daily_att_2(functional.tanh(daily_att_1))

            # Attention Weights and Embedding
            daily_A = functional.softmax(daily_att_2.t())
            daily_M = torch.mm(daily_A, daily_news)
            daily_BM[i, :] = daily_M.view(-1)
            daily_weights[i] = daily_A

        # MLP block for Classfier Feature
        MLPhidden = self.MLP(daily_BM)
        decoded = self.decoder(functional.relu(MLPhidden))

        return decoded, hidden, news_weights, daily_weights
                


# the self-attention model, 在基线模型中引入self-attention机制
class SelfModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, *input):
        pass




# the stock info model, 在基线模型中引入股票交易信息
class StockModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, *input):
        pass



# the self-attention and stock info model, 在基线模型中引入self-attention机制和股票交易信息
class SelfStockModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, *input):
        pass