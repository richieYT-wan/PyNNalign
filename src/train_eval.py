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

from torch.utils.data import DataLoader
from src.datasets import NNAlignDataset
from src.utils import get_motif
from src.torch_utils import save_checkpoint, load_checkpoint
from src.metrics import get_metrics


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


def predict_model(model, dataset: NNAlignDataset, dataloader: torch.utils.data.DataLoader):
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
    seq_col, window_size = dataset.seq_col, dataset.window_size
    df['motif'] = df.apply(get_motif, seq_col=seq_col, window_size=window_size, axis=1)
    return df


def train_eval_loops(n_epochs, tolerance, model, criterion, optimizer, train_dataset, train_loader, valid_loader,
                     checkpoint_filename, outdir, burn_in: int = None):
    """ Trains and validates a model over n_epochs, then reloads the best checkpoint


    Args:
        n_epochs:
        tolerance:
        model:
        criterion:
        optimizer:
        train_dataset: Torch Dataset, is here so we can do standardize & burn-in periods
        train_loader:
        valid_loader:
        checkpoint_filename:
        outdir:

    Returns:
        model
        train_metrics
        valid_metrics
        train_losses
        valid_losses
        best_epoch
        best_val_loss
        best_val_auc
    """

    if any([(hasattr(child, 'standardizer') or hasattr(child, 'ef_standardizer')) for child in [model.children()]+[model]]):
        # TODO: Not sure about this workaround (works the same as in above with *data & pop(-1))
        xs = train_dataset[:][:-1]
        model.fit_standardizer(*xs)

    if burn_in is not None:  # doing burn-in if it's not None
        print('Doing burn-in period')
        train_dataset.burn_in(True)
        for e in tqdm(range(0, burn_in), desc='Burn-in period'):
            _, _ = train_model_step(model, criterion, optimizer, train_loader)
        train_dataset.burn_in(False)

    print(f'Starting {n_epochs} training cycles')
    # Pre-saving the model at the very start because some bugged partitions
    # would have terrible performance and never save for very short debug runs.
    save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir)
    # Actual runs
    train_metrics, valid_metrics, train_losses, valid_losses = [], [], [], []
    best_val_loss, best_val_auc, best_epoch = 1000, 0.5, 1
    for e in tqdm(range(1, n_epochs + 1), desc='epochs', leave=False):
        train_loss, train_metric = train_model_step(model, criterion, optimizer, train_loader)
        valid_loss, valid_metric = eval_model_step(model, criterion, valid_loader)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if e % (n_epochs // 10) == 0:
            tqdm.write(f'\nEpoch {e}: train loss, AUC:\t{train_loss:.4f},\t{train_metric["auc"]:.3f}')
            tqdm.write(f'Epoch {e}: valid loss, AUC:\t{valid_loss:.4f},\t{valid_metric["auc"]:.3f}')

        # Doesn't allow saving the very first model as sometimes it gets stuck in a random state that has good
        # performance for whatever reasons
        if e > 1 and ((valid_loss <= best_val_loss + tolerance and valid_metric['auc'] > best_val_auc)\
                      or valid_metric['auc'] > best_val_auc):
            best_epoch = e
            best_val_loss = valid_loss
            best_val_auc = valid_metric['auc']
            save_checkpoint(model, filename=checkpoint_filename, dir_path=outdir)

    print(f'End of training cycles')
    print(f'Best train loss:\t{min(train_losses):.3e}, best train AUC:\t{max([x["auc"] for x in train_metrics])}')
    print(f'Best valid epoch: {best_epoch}')
    print(f'Best valid loss :\t{best_val_loss:.3e}, best valid AUC:\t{best_val_auc}')
    print(f'Reloaded best model at {os.path.abspath(os.path.join(outdir, checkpoint_filename))}')
    model = load_checkpoint(model, checkpoint_filename, outdir)
    return model, train_metrics, valid_metrics, train_losses, valid_losses, best_epoch, best_val_loss, best_val_auc
