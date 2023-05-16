from abc import ABC
from collections import OrderedDict
from typing import Union
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class NetParent(nn.Module):
    """
    Mostly a QOL superclass
    Creates a parent class that has reset_parameters implemented and .device
    so I don't have to re-write it to each child class and can just inherit it
    """

    def __init__(self):
        super(NetParent, self).__init__()
        # device is cpu by default
        self.device = 'cpu'

    def forward(self,x):
        raise NotImplementedError

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight.data)

    @staticmethod
    def reset_weight(layer):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    def reset_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for child in self.children():
            if hasattr(child, 'children'):
                for sublayer in child.children():
                    self.reset_weight(sublayer)
            if hasattr(child, 'reset_parameters'):
                self.reset_weight(child)

    def to(self, device):
        # Work around, so we can get model.device for all NetParent
        #
        super(NetParent, self).to(device)
        self.device = device


class Standardizer(nn.Module):
    def __init__(self):
        super(Standardizer, self).__init__()
        self.mu = 0
        self.sigma = 1
        self.fitted = False

    def fit(self, x_train):
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        self.mu = x_train.mean(axis=0)
        self.sigma = x_train.std(axis=0)
        # Fix issues with sigma=0 that would cause a division by 0 and return NaNs
        self.sigma[torch.where(self.sigma==0)] = 1e-12
        self.fitted = True

    def forward(self, x):
        assert self.fitted, 'Standardizer has not been fitted. Please fit to x_train'
        return (x - self.mu) / self.sigma

    def reset_parameters(self, **kwargs):
        self.mu = 0
        self.sigma = 0
        self.fitted = False


class NNAlign(NetParent):
    """
    This just runs the forward loop and selects the best loss.
    The inputs should be split in the ExpandDataset class with the unfold/transpose/reshape/flatten etc.
    """
    def __init__(self, n_hidden, window_size,
                 activation = nn.ReLU(),
                 dropout=0.0, indel=False):
        super(NNAlign, self).__init__()
        self.matrix_dim = 21 if indel else 20
        self.window_size = window_size
        self.in_layer = nn.Linear(self.window_size * self.matrix_dim, n_hidden)
        self.out_layer = nn.Linear(n_hidden, 1)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.act = activation

    def forward(self, x):
        x = self.dropout(self.in_layer(x))
        # Switch submer-feature dim to batchnorm then switch again to get preds
        # x = self.bn1(x.transpose(1,2)).transpose(1,2)
        x = self.act(x)
        x = self.out_layer(x)
        return x


class FFN(NetParent):
    def __init__(self, n_in=21, n_hidden=32, n_layers=1, act=nn.ReLU(), dropout=0.0):
        super(FFN, self).__init__()
        self.in_layer = nn.Linear(n_in, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.activation = act
        hidden_layers = [nn.Linear(n_hidden, n_hidden), self.dropout, self.activation] * n_layers
        self.hidden = nn.Sequential(*hidden_layers)
        # Either use Softmax with 2D output or Sigmoid with 1D output
        self.out_layer = nn.Linear(n_hidden, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.flatten(start_dim=1, end_dim=2)
        x = self.activation(self.in_layer(x))
        x = self.hidden(x)
        out = F.sigmoid(self.out_layer(x))
        return out

    def reset_parameters(self, **kwargs):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                try:
                    child.reset_parameters(**kwargs)
                except:
                    print('here xd', child)


class FFNetPipeline(NetParent):
    def __init__(self, n_in=21, n_hidden=32, n_layers=1, act=nn.ReLU(), dropout=0.3):
        super(FFNetPipeline, self).__init__()
        self.standardizer = Standardizer()
        self.input_length = n_in
        self.ffn = FFN(n_in, n_hidden, n_layers, act, dropout)

    def forward(self, x):
        # Need to do self.standardizer.fit() somewhere in the nested_kcv function
        x = self.standardizer(x)
        x = self.ffn(x)
        return x

    def fit_standardizer(self, x):
        assert self.training, 'Must be in training mode to fit!'
        self.standardizer.fit(x)

    def reset_parameters(self, **kwargs):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                try:
                    child.reset_parameters(**kwargs)
                except:
                    print('here xd', child)
