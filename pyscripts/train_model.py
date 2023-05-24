from src.data_processing import encode_batch
from src.utils import str2bool, pkl_dump, pkl_load, mkdirs
from src.models import NNAlign
from src.train_eval import train_model_step, eval_model_step
from src.datasets import get_NNAlign_dataloader
import pandas as pd
import os, sys
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F

import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a NNAlign model ')

    parser.add_argument('-tf', '--train_file', dest='train', required=True,
                        type=str, help='filename of the train_file, w/ extension & full path' \
                                       'ex: /path/to/file/train.csv')
    parser.add_argument('-vf', '--valid_file', dest='valid', required=True,
                        type=str, help='filename of the valid_file, w/ extension & full path' \
                                       'ex: /path/to/file/valid.csv')
    parser.add_argument('-x', '--seq_col', dest='seq_col', default='Sequence', type=str, required=True,
                        help='Name of the column containing sequences (inputs)')
    parser.add_argument('-y', '--target_col', dest='target_col', default='target', type=str, required=False,
                        help='Name of the column containing sequences (inputs)')

    parser.add_argument('-o', '--outdir', dest='outdir', required=True,
                        type=str, default='../output/', help='Where to save results')
    parser.add_argument('-nh', '--n_hidden', dest='n_hidden', required=True,
                        type=int, help='Number of hidden units')
    parser.add_argument('-std', '--standardize', dest='standardize', type=str2bool, required=True,
                        help='Whether to include standardization (True/False)')
    parser.add_argument('-bn', '--batchnorm', dest='batchnorm', type=str2bool, required=True,
                        help='Whether to add BatchNorm to the model (True/False)')
    parser.add_argument('-do', '--dropout', dest='dropout', type=float, default=0.0, required=False,
                        help='Whether to add DropOut to the model (p in float e[0,1], default = 0.0)')
    parser.add_argument('-k', '--window_size', dest='window_size', type=int, default=6, required=False,
                        help='Window size for sub-mers selection (default = 6)')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50L0, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-ml', '--max_len', dest='max_len', type=int, required=True,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-pad', '--pad_scale', dest='pad_scale', type=float, default=None, required=False,
                        help='Number with which to pad the inputs if needed; '\
                             'Default behaviour is 0 if onehot, -12 is BLOSUM')
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=1e-4, required=False,
                        help='Learning rate for the optimizer')
    parser.add_argument('-wd', '--weight_decay', dest='wd', type=float, default=1e-2, required=False,
                        help='Weight decay for the optimizer')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=128, required=False,
                        help='Batch size for mini-batch optimization')
    parser.add_argument('-ne', '--n_epochs', dest='n_epochs', type=int, default=500, required=False,
                        help='Number of epochs to train')
    return parser.parse_args()


def main():
    # I like dictionary for args :-)
    args = vars(args_parser())
    print(args)
    train_df = pd.read_csv(args['train'])
    valid_df = pd.read_csv(args['valid'])
    # TODO: For now we are doing like this because we don't care about other activations, singlepass, indels
    # Def params so it's ✨tidy ✨
    model_keys = ['n_hidden', 'window_size', 'batchnorm', 'dropout', 'standardize']
    dataset_keys = ['max_len', 'window_size', 'encoding', 'seq_col', 'target_col', 'pad_scale', 'batch_size']
    model_params = {k: args[k] for k in model_keys}
    dataset_params = {k: args[k] for k in dataset_keys}
    optim_params = {'lr': args['lr'], 'wd': args['wd']}

    # instantiate objects
    model = NNAlign(**model_params, singlepass=True, indel=False)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), **optim_params)
    train_loader, train_dataset = get_NNAlign_dataloader(train_df, return_dataset=True, indel=False, **dataset_params)
    valid_loader, valid_dataset = get_NNAlign_dataloader(valid_df, return_dataset=True, indel=False, **dataset_params)

    train_losses, valid_losses = [], []
    train_preds, valid_preds = [], []
    for e in range(1, args['n_epochs'] + 1):
        pass

if __name__ == '__main__':
    main()
