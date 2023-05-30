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

    def forward(self, x_tensor):
        """
        Args:
            x:
            x_mask: x_mask here exists for compatibility purposes


        Returns:

        """

        return x_tensor

    def fit(self, x_tensor, x_mask):
        """
        Args:
            x:
            x_mask: x_mask here exists for compatibility purposes


        Returns:

        """
        self.fitted=True
        return x_tensor


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

    def predict_logits(self, x_tensor:torch.Tensor, x_mask:torch.Tensor):
        """ QOL method to return the predictions without Sigmoid + return the indices
        To be used elsewhere down the line (in EF model)

        Args:
            x_tensor:
            x_mask:

        Returns:

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
            z = torch.gather(z, 1, max_idx).squeeze(1)
            return z, max_idx


class NNAlign(NetParent):
    def __init__(self, n_hidden, window_size, activation=nn.SELU(), batchnorm=False, dropout=0.0, indel=False,
                 standardize=True, **kwargs):
        super(NNAlign, self).__init__()
        self.nnalign = NNAlignSinglePass(n_hidden, window_size, activation, batchnorm, dropout, indel)
        self.standardizer = Standardizer() if standardize else StdBypass()
        # Save here to make reloading a model potentially easier
        self.init_params = {'n_hidden': n_hidden, 'window_size': window_size, 'activation': activation,
                            'batchnorm': batchnorm, 'dropout': dropout, 'indel': indel,
                            'standardize': standardize}

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

    def predict_logits(self, x_tensor:torch.Tensor, x_mask: torch.Tensor):
        with torch.no_grad():
            x_tensor = self.standardizer(x_tensor)
            x_tensor, max_idx = self.nnalign.predict_logits(x_tensor, x_mask)
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


class NNAlignEF(NetParent):
    """ EF == ExtraFeatures
    TODO: Currently assumes that I need an extra in_layer + an extra out_layer
          Could also be changed to take a single extra layer of nn.Linear(1+n_extrafeatures, 1)
          That takes as input the logits from NNAlign + the extra features and directly returns a score without 2 layers.
          Can maybe write another class EFModel that just takes the ef_xx part here
    """
    def __init__(self, n_hidden, window_size, activation=nn.SELU(), batchnorm=False, dropout=0.0,
                 indel=False, standardize=True,
                 n_extrafeatures=0, n_hidden_ef=5, activation_ef=nn.SELU(), batchnorm_ef=False, dropout_ef=0.0,
                 **kwargs):
        super(NNAlignEF, self).__init__()
        # NNAlign part
        self.nnalign_model = NNAlign(n_hidden, window_size, activation, batchnorm, dropout, indel, standardize)
        # Extra layer part
        self.in_dim = n_extrafeatures + 1 # +1 because that's the dimension of the logit scores returned by NNAlign
        self.ef_standardizer = Standardizer() if standardize else StdBypass()
        self.ef_inlayer = nn.Linear(self.in_dim, n_hidden_ef)
        self.ef_outlayer = nn.Linear(n_hidden_ef, 1)
        self.ef_act = activation_ef
        self.ef_dropout = nn.Dropout(dropout_ef)
        self.ef_batchnorm = batchnorm_ef
        if batchnorm_ef:
            self.ef_bn1 = nn.BatchNorm1d(n_hidden_ef)

        self.init_params = {'n_hidden': n_hidden, 'window_size': window_size, 'activation': activation,
                            'batchnorm': batchnorm, 'dropout': dropout, 'indel': indel, 'standardize': standardize,
                            'n_extrafeatures':n_extrafeatures, 'n_hidden_ef':n_hidden_ef, 'activation_ef':activation_ef,
                            'batchnorm_ef':batchnorm_ef, 'dropout_ef':dropout_ef}

    def fit_standardizer(self, x_tensor:torch.Tensor, x_mask:torch.Tensor, x_features:torch.Tensor):
        self.nnalign_model.fit_standardizer(x_tensor, x_mask)
        self.ef_standardizer.fit(x_features)

    def forward(self, x_tensor:torch.Tensor, x_mask:torch.Tensor, x_features:torch.Tensor):
        # NNAlign part
        z = self.nnalign_model(x_tensor, x_mask)
        # Extra features part, standardizes, concat
        x_features = self.ef_standardizer(x_features)
        z = torch.cat([z, x_features], dim=1)
        # Standard NN stuff for the extra layers
        z = self.ef_inlayer(z)
        if self.ef_batchnorm:
            z = self.ef_bn1(z)
        z = self.ef_act(self.ef_dropout(z))
        # Returning logits
        z = self.ef_outlayer(z)
        return z

    def predict(self, x_tensor:torch.Tensor, x_mask:torch.Tensor, x_features:torch.Tensor):
        """ TODO: This is a bit convoluted and could be reworked to be more efficient
                  Would probly require to modify the other classes a bit though

        Args:
            x_tensor:
            x_mask:
            x_features:

        Returns:

        """
        with torch.no_grad():
            # Return logits from nnalign model + max idx
            z, max_idx = self.nnalign_model.predict_logits(x_tensor, x_mask)

            # Standard NN stuff for the extra layers
            x_features = self.ef_standardizer(x_features)
            z = torch.cat([z, x_features], dim=1)
            z = self.ef_inlayer(z)
            if self.ef_batchnorm:
                z = self.ef_bn1(z)
            z = self.ef_act(self.ef_dropout(z))
            # Returning probs [0, 1]
            z = F.sigmoid(self.ef_outlayer(z))
            return z, max_idx

    def state_dict(self, **kwargs):
        state_dict = super(NNAlignEF, self).state_dict()
        state_dict['nnalign_model'] = self.nnalign_model.state_dict()
        state_dict['ef_standardizer'] = self.ef_standardizer.state_dict()
        state_dict['init_params'] = self.init_params

    def load_state_dict(self, state_dict, **kwargs):
        self.nnalign_model.load_state_dict(state_dict['nnalign_model'])
        self.ef_standardizer.load_state_dict(state_dict['ef_standardizer'])
        self.init_params = state_dict['init_params']




