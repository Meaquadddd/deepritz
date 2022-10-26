from audioop import add
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FC_DNN(nn.Sequential):
    def __init__(self, dim_in, dim_out, dim_hidden, layers_hidden, act='tanh', 
        xavier_init=True):
        super(FC_DNN,self).__init__()

        self.add_module('fc0', nn.Linear(dim_in, dim_hidden, bias=None))
        self.add_module('act0', nn.Tanh())
        for i in range(1, layers_hidden):
            self.add_module('fc{}'.format(i), nn.Linear(dim_hidden, dim_hidden, bias=True))
            if act == 'tanh':
                self.add_module('act{}'.format(i), nn.Tanh())
            elif act == 'relu':
                self.add_module('act{}'.format(i), nn.ReLU())
            else:
                raise ValueError(f'unknown activation function: {act}')

        self.add_module('fc{}'.format(layers_hidden), nn.Linear(dim_hidden, dim_out))
        if xavier_init:
            self.init_xavier()

    def init_xavier(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_normal_(param)

    


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    else:
        raise ValueError('Unknown activation function')


class _ResLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, act='tanh'):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=True)
        self.fc2 = nn.Linear(dim_hidden, dim_out, bias=True)
        if act == 'tanh':
            self.act = F.tanh 
        elif act == 'relu':
            self.act = F.relu

    def forward(self, x):
        # pre-activation
        res = x
        out = self.fc1(self.act(x))
        out = self.fc2(self.act(out))
        return res + out

class Res_FC_DNN(nn.Sequential):
    '''
    残差链接的网络块
    '''
    def __init__(self, dim_in, dim_out, dim_hidden, res_layers, act='tanh'):
        super().__init__()
        self.add_module('fc0', nn.Linear(dim_in, dim_hidden, bias=None))

        for i in range(res_layers):
            reslayer = _ResLayer(dim_hidden, dim_hidden, dim_hidden, act=act)
            self.add_module(f'reslayer{i+1}', reslayer)
        
        self.add_module('act_last', activation(act))
        self.add_module('fc_last', nn.Linear(dim_hidden, dim_out, bias=True))