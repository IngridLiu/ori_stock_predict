import torch.nn as nn
from torch.autograd import Variable



# the basic model, 基线模型
class BasicModel(nn.Module):
    def __init__(self, N, dinp, nhid, nlayers, mlp_nhid):
        ''''''

        super(BasicModel, self).__init__()

        # daily news attention layer

        # RNN layer

        # daily attention layer

        # final mlp layer

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        return(Variable(weight.new(self.nlayers * 2, batch_size, self.nhid).zeros()),
               Variable(weight.new(self.nlayers * 2, batch_size, self.nhid).zeros()))

    def forward(self, *input):
        pass




# the self-attention model, 在基线模型中引入attention机制
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