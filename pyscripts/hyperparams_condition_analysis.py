import pandas as pd
from tqdm.auto import tqdm
import os, sys
from os import path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch import optim
from torch import nn
from torch.utils.data import SequentialSampler, RandomSampler
from datetime import datetime as dt
from src.metrics import get_metrics
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_loss_aucs
from src.torch_utils import save_checkpoint, load_checkpoint
from src.bootstrap import bootstrap_eval
from src.models import NNAlignEF
from src.train_eval import train_model_step, eval_model_step, predict_model, train_eval_loops
from sklearn.model_selection import train_test_split
from src.datasets import get_NNAlign_dataloader
from matplotlib import pyplot as plt
import seaborn as sns

import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a NNAlign model ')
    """
    Data processing args
    """
    parser.add_argument('-d', '--dir', dest='dir', required=True, type=str,
                        help='Path that contains all KCV subfolders for a given condition')
    parser.add_argument('-t', '--test', dest='test', required=True, type=str,
                        help='Whether a test file was also included, otherwise only run analysis for validation')
    parser.add_argument('-x', '--pred_col', dest='pred_col', default='pred', type=str, required=False,
                        help='Name of the column containing sequences (inputs)')
    parser.add_argument('-y', '--target_col', dest='target_col', default='target', type=str, required=False,
                        help='Name of the column containing sequences (inputs)')
    parser.add_argument('-b', '--bootstrap', dest='bootstrap', default=None, type=int, required=False,
                        help='Whether to bootstrap. -b should'\
                             ' either be left untouched for no bootstrapping, or an int for the number of rounds.')
    return parser.parse_args()


"""
Goes to a given path and does analysis on the results/predictions returned
"""
def get_fold(string, filename):
    """
    Quick & dirty code that I'm sure won't bite me in the ass later.
    Args:
        string:
        filename:

    Returns:
        Gets the kcv fold
    """
    return int(string.split('_'+filename)[0].split('_f')[1])


def get_hps(filename):
    keys = ['encoding', 'pad', 'main_nh', 'std', 'main_bn', 'main_drop',
            'window_size', 'extra_nh', 'extra_bn', 'extra_drop', 'lr', 'wd', 'batch_size',
            'features']
    dtypes = {'encoding':str,
              'pad':int,
              'std': bool,
              'main_nh':int,
              'main_bn':bool,
              'main_drop':float,
              'window_size':int,
              'extra_nh':int,
              'extra_bn':bool,
              'extra_drop':float,
              'lr':float,
              'wd':float,
              'batch_size':int,
              'features':str}
    params = [x.replace('xx','-').replace('zp','0.').replace('XX','_') for x in filename.split('_')]
    params = [x.capitalize() if x in ['true', 'false'] else x for x in params]
    return {k:dtypes[k](v) for k,v in zip(keys,params)}



def main():
    start = dt.now()
    # I like dictionary for args :-)
    args = vars(args_parser())
    # This is also the main filename for all subsequent subdirectories/file
    # Use this to recover the kwargs / params, or use the kwargs that are written to args.txt?
    maindir = args['dir']
    pcol, tcol = args['x'], args['y']
    subdirs = [x for x in os.listdir(args['dir']) if path.isdir(x)]
    valid_preds, test_preds = [], []
    hyperparams = pd.DataFrame(get_hps(maindir), index=[0])

    for i, subdir in enumerate(subdirs):
        fold = get_fold(os.listdir(maindir+subdir)[0], maindir)
        valid = pd.read_csv(
            f"{maindir}{subdir}/{list(filter(lambda x: 'valid_pred' in x, os.listdir(maindir+subdir)))[0]}").assign(fold=fold)
        valid_preds.append(valid)
        if args['test']:
            test = pd.read_csv(
            f"{maindir}{subdir}/{list(filter(lambda x: 'test_pred' in x, os.listdir(maindir + subdir)))[0]}").assign(fold=fold)
            test_preds.append(test)

    valid_preds = pd.concat(valid_preds)
    valid_scores, valid_labels = valid_preds[pcol].values, valid_preds[tcol].values
    # TODO: Make per-fold, concat, mean metrics ; Also include ToBoot or NotToBoot
    if args['test']:
        test_preds = pd.concat(test_preds)
        test_scores, test_labels = test_preds[pcol].values, test_preds[tcol].values




if __name__=='__main__':
    main()