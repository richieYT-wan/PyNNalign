from abc import ABC
from collections import OrderedDict
from typing import Union
import numpy as np
import sklearn
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

    def forward(self, x):
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
        self.dimensions = None

    def fit(self, x):
        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        with torch.no_grad():
            x = self.view_3d_to_2d(x)
            self.mu = x.mean(axis=0)
            self.sigma = x.std(axis=0)
            # Fix issues with sigma=0 that would cause a division by 0 and return NaNs
            self.sigma[torch.where(self.sigma == 0)] = 1e-12
            self.fitted = True

    def forward(self, x):
        assert self.fitted, 'Standardizer has not been fitted. Please fit to x_train'
        with torch.no_grad():
            # Flatten to 2d if needed
            x = (self.view_3d_to_2d(x) - self.mu) / self.sigma
            # Return to 3d if needed
            return self.view_2d_to_3d(x)

    def recover(self, x):
        assert self.fitted, 'Standardizer has not been fitted. Please fit to x_train'
        with torch.no_grad():
            # Flatten to 2d if needed
            x = self.view_3d_to_2d(x)
            # Return to original scale by multiplying with sigma and adding mu
            x = x * self.sigma + self.mu
            # Return to 3d if needed
            return self.view_2d_to_3d(x)

    def reset_parameters(self, **kwargs):
        with torch.no_grad():
            self.mu = 0
            self.sigma = 0
            self.fitted = False

    def view_3d_to_2d(self, x):
        with torch.no_grad():
            if len(x.shape) == 3:
                self.dimensions = (x.shape[0], x.shape[1], x.shape[2])
                return x.view(-1, x.shape[2])
            else:
                return x

    def view_2d_to_3d(self, x):
        with torch.no_grad():
            if len(x.shape) == 2 and self.dimensions is not None:
                return x.view(self.dimensions[0], self.dimensions[1], self.dimensions[2])
            else:
                return x


class NNAlign(NetParent):
    """TODO : DEPRECATED
    This just runs the forward loop and selects the best loss.
    The inputs should be split in the ExpandDataset class with the unfold/transpose/reshape/flatten etc.
    TODO: implement 2 versions of it, one with the double forward pass, and one with a single + with torch nograd selection to get the argmax
        If the results are the same, then just keep the network with a single forward pass
        + implement crossvalidation + nested crossvalidation
    """

    def __init__(self, n_hidden, window_size,
                 activation=nn.ReLU(), batchnorm=False,
                 dropout=0.0, indel=False):
        super(NNAlign, self).__init__()
        self.matrix_dim = 21 if indel else 20
        self.window_size = window_size
        self.n_hidden = n_hidden
        self.in_layer = nn.Linear(self.window_size * self.matrix_dim, n_hidden)
        self.out_layer = nn.Linear(n_hidden, 1)
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.act = activation

    def forward(self, x):
        """
        Here, do a double forward pass to be sure not to compute the gradient when doing any of the steps
        for the best submer selection
        :param x:
        :return:
        """

        # FIRST FORWARD PASS: best scoring selection, with no grad
        with torch.no_grad():
            z = self.in_layer(x)  # Inlayer
            if self.batchnorm:
                z = self.bn1(z.view(x.shape[0] * x.shape[1], self.n_hidden)).view(-1, x.shape[1], self.n_hidden)
            z = self.dropout(z)
            z = self.act(z)
            z = self.out_layer(z)  # Out Layer for prediction
            # NNAlign selecting the max score here
            max_idx = z.argmax(dim=1).unsqueeze(1).expand(-1, -1, self.window_size * self.matrix_dim)
            x = torch.gather(x, 1, max_idx).squeeze(1)  # Indexing the best submers

        # SECOND FORWARD PASS: normal prediction, with gradient
        # Doing a simple no batchnorm no dropout version for now ;

        # Simple fwd pass : in_layer -> BN -> DropOut -> Activation
        x = self.in_layer(x)
        if self.batchnorm:
            x = self.bn1(x)
        x = self.act(self.dropout(x))
        x = self.out_layer(x)
        return x


class NNAlignSinglePass(NetParent):
    """
    NNAlign implementation with a single forward pass where best score selection + indexing is done in one pass.
    """
    def __init__(self, n_hidden, window_size,
                 activation=nn.ReLU(), batchnorm=False,
                 dropout=0.0, indel=False):
        super(NNAlignSinglePass, self).__init__()
        self.matrix_dim = 21 if indel else 20
        self.window_size = window_size
        self.n_hidden = n_hidden
        self.in_layer = nn.Linear(self.window_size * self.matrix_dim, n_hidden)
        self.out_layer = nn.Linear(n_hidden, 1)
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.act = activation

    def forward(self, x:torch.Tensor):
        """
        Single forward pass for layers + best score selection without w/o grad
        Args:
            x:

        Returns:

        """
        # FIRST FORWARD PASS: best scoring selection, with no grad
        z = self.in_layer(x)  # Inlayer
        # Flip dimensions to allow for batchnorm then flip back
        if self.batchnorm:
            z = self.bn1(z.view(x.shape[0] * x.shape[1], self.n_hidden)).view(-1, x.shape[1], self.n_hidden)
        z = self.dropout(z)
        z = self.act(z)
        z = self.out_layer(z)  # Out Layer for prediction
        # NNAlign selecting the max score here
        with torch.no_grad():
            max_idx = z.argmax(dim=1).unsqueeze(1)
        z = torch.gather(z, 1, max_idx).squeeze(1)  # Indexing the best submers
        return z

    def predict(self, x:torch.Tensor):
        """Works like forward but also returns the index (for the motif selection/return)

        This should be done with torch no_grad as this shouldn't be used during/for training
        Also here does the sigmoid to return scores within [0, 1] on Z
        TODO: Check whether this fct should also return window size.
        Args:
            x: torch.Tensor,

        Returns:
            z: torch.Tensor, the best scoring K-mer for each of the input in X
            max_idx: torch.Tensor, the best indices corresponding to the best K-mer,
                     used to find the predicted core
        """
        with torch.no_grad():
            z = self.in_layer(x)
            if self.batchnorm:
                z = self.bn1(z.view(x.shape[0] * x.shape[1], self.n_hidden)).view(-1, x.shape[1], self.n_hidden)
            z = self.act(self.dropout(z))
            z = F.sigmoid(self.out_layer(z))
            max_idx = z.argmax(dim=1).unsqueeze(1)
            z = torch.gather(z, 1, max_idx).squeeze(1)
            return z, max_idx

class NNAlignWrapper(NetParent):
    def __init__(self, n_hidden, window_size, activation=nn.ReLU(),
                 batchnorm=False, dropout=0.0, indel=False, singlepass=True):
        super(NNAlignWrapper, self).__init__()
        NN = {False: NNAlign,
              True: NNAlignSinglePass}
        self.nnalign = NN[singlepass](n_hidden, window_size, activation, batchnorm, dropout, indel)
        self.standardizer = Standardizer()

    def fit_standardizer(self, x:torch.Tensor):
        assert self.training, 'Must be in training mode to fit!'
        with torch.no_grad():
            self.standardizer.fit(x)

    def forward(self, x:torch.Tensor):
        with torch.no_grad():
            x = self.standardizer(x)
        x = self.nnalign(x)
        return x

    def predict(self, x:torch.Tensor):
        with torch.no_grad():
            x = self.standardizer(x)
            x, max_idx = self.nnalign.predict(x)
            return x, max_idx

    def reset_parameters(self, **kwargs):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                try:
                    child.reset_parameters(**kwargs)
                except:
                    print('here xd', child)

