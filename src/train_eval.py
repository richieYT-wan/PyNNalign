import copy
import multiprocessing
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from joblib import Parallel, delayed
from functools import partial
from tqdm.auto import tqdm
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from src.utils import make_chunks
from src.torch_utils import set_mode, set_device
from src.data_processing import get_dataset, assert_encoding_kwargs
from src.metrics import get_metrics, get_predictions, get_mean_roc_curve


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=1e-6, name='checkpoint'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.prev_best_score = np.Inf
        self.delta = delta
        self.path = f'{name}.pt'

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            # This condition works for AUC ; checks that the AUC is below the best AUC +/- some delta
            if score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Prev best score: ({self.prev_best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.prev_best_score = score


def invoke(early_stopping, loss, model, implement=False):
    if implement:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            return True
    else:
        return False


def train_model_step(model, criterion, optimizer, train_loader):
    """
    230525: Updated train_loader behaviour. Now returns x_tensor, x_mask, y for each idx, used to remove padded positions
            in the forward of NNAlign. vvv Change signature below
    Args:
        model:
        criterion:
        optimizer:
        train_loader:

    Returns:

    """
    assert type(train_loader.sampler) == torch.utils.data.RandomSampler, 'TrainLoader should use RandomSampler!'
    model.train()
    train_loss = 0
    y_scores, y_true = [], []
    # Here, workaround so that the same fct can pass different number of arguments to the model
    # e.g. to accomodate for an extra x_feature tensor if returned by train_loader
    for data in train_loader:
        y_train = data.pop(-1)
        output = model(*data)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_true.append(y_train)
        y_scores.append(F.sigmoid(output))
        train_loss += loss.item() * y_train.shape[0]

    # Concatenate the y_pred & y_true tensors and compute metrics
    y_scores, y_true = torch.cat(y_scores), torch.cat(y_true)
    train_metrics = get_metrics(y_true, y_scores, threshold=0.5, reduced=True)
    # Normalizes to loss per batch
    train_loss /= len(train_loader.dataset)
    return train_loss, train_metrics


def eval_model_step(model, criterion, valid_loader):
    model.eval()
    # disables gradient logging
    valid_loss = 0
    y_scores, y_true = [], []
    with torch.no_grad():
        # Same workaround as above
        for data in valid_loader:
            y_valid = data.pop(-1)
            output = model(*data)
            loss = criterion(output, y_valid)
            y_true.append(y_valid)
            y_scores.append(F.sigmoid(output))
            valid_loss += loss.item() * y_valid.shape[0]
    # Concatenate the y_pred & y_true tensors and compute metrics
    y_scores, y_true = torch.cat(y_scores), torch.cat(y_true)
    valid_metrics = get_metrics(y_true, y_scores, threshold=0.5, reduced=True)
    # Normalizes to loss per batch
    valid_loss /= len(valid_loader.dataset)
    return valid_loss, valid_metrics


def predict_model(model, dataset: torch.utils.data.Dataset, dataloader: torch.utils.data.DataLoader):
    assert type(dataloader.sampler) == torch.utils.data.SequentialSampler, \
        'Test/Valid loader MUST use SequentialSampler!'

    assert hasattr(dataset, 'df'), 'Not DF found for this dataset!'
    model.eval()
    df = dataset.df.reset_index(drop=True).copy()
    # indices = range(len(df))
    # idx_batches = make_chunks(indices, batch_size)
    predictions, best_indices, ys = [], [], []
    # HERE, MUST ENSURE WE USE
    with torch.no_grad():
        # Same workaround as above
        for data in dataloader:
            y = data.pop(-1)
            preds, core_idx = model.predict(*data)

            predictions.append(preds)
            best_indices.append(core_idx)
            ys.append(y)
    predictions = torch.cat(predictions).detach().cpu().numpy().flatten()
    best_indices = torch.cat(best_indices).detach().cpu().numpy().flatten()
    ys = torch.cat(ys).detach().cpu().numpy().flatten()
    df['pred'] = predictions
    df['core_start_index'] = best_indices
    df['label'] = ys
    return df
