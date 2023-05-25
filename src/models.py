from abc import ABC, abstractmethod, abstractstaticmethod
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

    def fit(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        """ Will consider the mask (padded position) and ignore them before computing the mean/std
        Args:
            x_tensor:
            x_mask:

        Returns:

        """

        assert self.training, 'Can not fit while in eval mode. Please set model to training mode'
        with torch.no_grad():
            # TODO: deprecated 3d to 2d here / 2d to 3d here, but will stay in forward.
            # x = self.view_3d_to_2d(x)
            # Updated version with masking
            masked_values = x_tensor * x_mask
            mu = (torch.sum(masked_values, dim=1) / torch.sum(x_mask, dim=1))
            sigma = (torch.sqrt(torch.sum((masked_values - mu.unsqueeze(1))**2, dim=1) / torch.sum(x_mask, dim=1)))
            self.mu = mu.mean(dim=0)
            self.sigma = sigma.mean(dim=0)
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

    def state_dict(self, **kwargs):
        """overwrites the state_dict with the custom attributes

        Returns: state_dict

        Args:
            **kwargs:
        """
        state_dict = super(Standardizer, self).state_dict()
        state_dict['mu'] = self.mu
        state_dict['sigma'] = self.sigma
        state_dict['fitted'] = self.fitted
        state_dict['dimensions'] = self.dimensions
        return state_dict

    def load_state_dict(self, state_dict, **kwargs):
        self.mu = state_dict['mu']
        self.sigma = state_dict['sigma']
        self.fitted = state_dict['fitted']
        self.dimensions = state_dict['dimensions']


class StdBypass(nn.Module):
    def __init__(self, **kwargs):
        super(StdBypass, self).__init__()
        self.requires_grad = False
        self.bypass = nn.Identity(**kwargs)
        self.fitted = False
        self.mu = 0
        self.sigma = 1

    def forward(self, x):
        return self.bypass(x)

    def fit(self, x, **kwargs):
        self.fitted=True
        return self.bypass(x)


class NNAlignSinglePass(NetParent):
    """
    NNAlign implementation with a single forward pass where best score selection + indexing is done in one pass.
    """

    def __init__(self, n_hidden, window_size,
                 activation, batchnorm=False,
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

    def forward(self, x_tensor: torch.Tensor, x_mask: torch.tensor):

        """
        Single forward pass for layers + best score selection without w/o grad
        Args:
            x_mask:
            x_tensor:

        Returns:

        """
        # FIRST FORWARD PASS: best scoring selection, with no grad
        z = self.in_layer(x_tensor)  # Inlayer
        # Flip dimensions to allow for batchnorm then flip back
        if self.batchnorm:
            z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden))\
                    .view(-1, x_tensor.shape[1], self.n_hidden)
        z = self.dropout(z)
        z = self.act(z)
        z = self.out_layer(z)  # Out Layer for prediction

        # NNAlign selecting the max score here
        with torch.no_grad():
            # Here, use sigmoid to set values to 0,1 before masking
            # only for index selection, Z will be returned as logits
            max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)

        z = torch.gather(z, 1, max_idx).squeeze(1)  # Indexing the best submers
        return z

    def predict(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        """Works like forward but also returns the index (for the motif selection/return)

        This should be done with torch no_grad as this shouldn't be used during/for training
        Also here does the sigmoid to return scores within [0, 1] on Z
        Args:
            x_tensor: torch.Tensor, the input tensor (i.e. encoded sequences)
            x_mask: torch.Tensor, to mask padded positions
        Returns:
            z: torch.Tensor, the best scoring K-mer for each of the input in X
            max_idx: torch.Tensor, the best indices corresponding to the best K-mer,
                     used to find the predicted core
        """
        with torch.no_grad():
            z = self.in_layer(x_tensor)
            if self.batchnorm:
                z = self.bn1(z.view(x_tensor.shape[0] * x_tensor.shape[1], self.n_hidden))\
                        .view(-1, x_tensor.shape[1], self.n_hidden)
            z = self.act(self.dropout(z))
            z = self.out_layer(z)
            # Do the same trick where the padded positions are removed prior to selecting index
            max_idx = torch.mul(F.sigmoid(z), x_mask).argmax(dim=1).unsqueeze(1)
            # Additionally run sigmoid on z so that it returns proba in range [0, 1]
            z = F.sigmoid(torch.gather(z, 1, max_idx).squeeze(1))
            return z, max_idx


class NNAlign(NetParent):
    def __init__(self, n_hidden, window_size, activation=nn.SELU(), batchnorm=False, dropout=0.0, indel=False,
                 standardize=True, **kwargs):
        super(NNAlign, self).__init__()
        # TODO:
        #  This is also deprecated, should just use single pass but leave it for now in case it's needed later

        self.nnalign = NNAlignSinglePass(n_hidden, window_size, activation, batchnorm, dropout, indel)
        self.standardizer = Standardizer() if standardize else StdBypass()
        # Save here to make reloading a model potentially easier
        self.init_params = {'n_hidden': n_hidden, 'window_size': window_size, 'activation': activation,
                            'batchnorm': batchnorm, 'dropout': dropout, 'indel': indel,
                            'standardizer': standardize}

    def fit_standardizer(self, x_tensor: torch.Tensor, x_mask):
        assert self.training, 'Must be in training mode to fit!'
        with torch.no_grad():
            self.standardizer.fit(x_tensor, x_mask)

    def forward(self, x_tensor: torch.Tensor, x_mask:torch.Tensor):
        with torch.no_grad():
            x_tensor = self.standardizer(x_tensor)
        x_tensor = self.nnalign(x_tensor, x_mask)
        return x_tensor

    def predict(self, x_tensor: torch.Tensor, x_mask: torch.Tensor):
        with torch.no_grad():
            x_tensor = self.standardizer(x_tensor)
            x_tensor, max_idx = self.nnalign.predict(x_tensor, x_mask)
            return x_tensor, max_idx

    def reset_parameters(self, **kwargs):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                try:
                    child.reset_parameters(**kwargs)
                except:
                    print('here xd', child)

    def state_dict(self, **kwargs):
        state_dict = super(NNAlign, self).state_dict()
        state_dict['nnalign'] = self.nnalign.state_dict()
        state_dict['standardizer'] = self.standardizer.state_dict()
        state_dict['init_params'] = self.init_params
        return state_dict

    def load_state_dict(self, state_dict, **kwargs):
        self.nnalign.load_state_dict(state_dict['nnalign'])
        self.standardizer.load_state_dict(state_dict['standardizer'])
        self.init_params = state_dict['init_params']
