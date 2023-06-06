import pandas as pd
from tqdm.auto import tqdm
import os, sys
from os import path

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from datetime import datetime as dt
from src.metrics import get_metrics
from functools import partial
from joblib import Parallel, delayed
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
    # parser.add_argument('-t', '--test', dest='test', required=True, type=str,
    #                     help='Whether a test file was also included, otherwise only run analysis for validation')
    parser.add_argument('-x', '--pred_col', dest='pred_col', default='pred', type=str, required=False,
                        help='Name of the column containing sequences (inputs)')
    parser.add_argument('-y', '--target_col', dest='target_col', default='target', type=str, required=False,
                        help='Name of the column containing sequences (inputs)')
    parser.add_argument('-nc', '--n_cores', dest='n_cores', default=1, type=int, required=True,
                        help='Number of jobs in parallel (1-40)')
    # parser.add_argument('-b', '--bootstrap', dest='bootstrap', default=None, type=int, required=False,
    #                     help='Whether to bootstrap. -b should' \
    #                          ' either be left untouched for no bootstrapping, or an int for the number of rounds.')
    return parser.parse_args()


"""
Goes to a given path and does analysis on the results/predictions returned
"""


def get_fold(args_txtfile):
    """
    Quick & dirty code that I'm sure won't bite me in the ass later.
    ## TODO: Actually, it's as dirty but slightly better to read the args txt file and parse the kcv from there
    Args:
        string:
        filename:

    Returns:
        Gets the kcv fold
    """
    # return int(string.split('_' + filename)[0].split('_f')[1])
    with open(args_txtfile, 'r') as f:
        return int([x.replace('fold: ', '').replace('\n', '') for x in f.readlines() if x.startswith('fold:')][0])


def get_hps(filename):
    keys = ['encoding', 'pad', 'main_nh', 'std', 'main_bn', 'main_drop',
            'window_size', 'extra_nh', 'extra_bn', 'extra_drop', 'lr', 'wd', 'batch_size',
            'features']
    dtypes = {'encoding': str,
              'pad': int,
              'std': bool,
              'main_nh': int,
              'main_bn': bool,
              'main_drop': float,
              'window_size': int,
              'extra_nh': int,
              'extra_bn': bool,
              'extra_drop': float,
              'lr': float,
              'wd': float,
              'batch_size': int,
              'features': str}
    params = [x.replace('xx', '-').replace('zp', '0.').replace('XX', '_').replace('/', '') for x in filename.split('_')]
    params = [x.capitalize() if x in ['true', 'false'] else x for x in params]
    return {k: dtypes[k](v) for k, v in zip(keys, params)}


def pipeline(fold_dir, args):
    """
    Assumes fold_dir is the main fold_dir containing X folds, e.g. also the "filename" that gives the hyperparams.
    i.e. fold_dir is /XXX/ in /path/to/output/hyperparameters_tuning/XXX/
    Should do the following:
        - Get the preds/performance on a per-fold basis
        - Get the performance on a mean prediction basis (for test)
        - Get the performance on a concat prediction (for valid+test)
        - Save the per fold perf to the `fold_dir`
        - return the mean&concat performance to make a big mega df?
    """

    maindir = args['dir'] + '/' if not args['dir'].endswith('/') else args['dir']
    fold_dir = fold_dir + '/' if not fold_dir.endswith('/') else fold_dir
    # Ex should be .../output/2306xx_hyperparams_tuning/
    pcol, tcol = args['pred_col'], args['target_col']
    subdirs = [x for x in os.listdir(maindir + fold_dir) if path.isdir(path.join(maindir + fold_dir, x))]
    valid_preds, test_preds = [], []
    hyperparams = get_hps(fold_dir)
    per_fold = []
    for subdir in subdirs:
        fullpath = maindir + fold_dir + subdir + '/'
        args_txtfile = next(filter(lambda x: 'args' in x and x.endswith('.txt'), os.listdir(fullpath)))
        fold = get_fold(args_txtfile=fullpath + args_txtfile)
        # Appending to do mean/concat things at the end
        valid = pd.read_csv(f"{fullpath}{next(filter(lambda x: 'valid_pred' in x, os.listdir(fullpath)))}").assign(
            fold=fold)
        valid_preds.append(valid)
        test = pd.read_csv(f"{fullpath}{next(filter(lambda x: 'test_pred' in x, os.listdir(fullpath)))}").assign(
            fold=fold)
        test_preds.append(test)

        valid_metrics = pd.DataFrame(get_metrics(valid[tcol], valid[pcol], threshold=0.5), index=[fold])
        valid_metrics.columns = [f'valid_{x}' for x in valid_metrics.columns]
        test_metrics = pd.DataFrame(get_metrics(test[tcol], test[pcol], threshold=0.5), index=[fold])
        test_metrics.columns = [f'test_{x}' for x in test_metrics.columns]
        per_fold.append(pd.concat([valid_metrics, test_metrics], axis=1).assign(fold=fold))

    per_fold, valid_preds, test_preds = pd.concat(per_fold), pd.concat(valid_preds), pd.concat(test_preds)
    # Adding mean perf
    per_fold = pd.concat([per_fold.sort_values('fold'), per_fold.mean(axis=0)], axis=1).replace(
        to_replace={'fold': 4.5}, value='average')
    # Adding HP
    per_fold = pd.concat([per_fold, pd.concat([pd.DataFrame(hyperparams, index=[i]) for i in range(len(per_fold))])],
                         axis=1)
    per_fold.to_csv(f'{maindir}{fold_dir}per_fold_results.csv', index=False)
    # Getting the mean & concat valid+test perf and returning to the mega df
    # Mean test:
    test_mean_preds = test_preds.groupby(['HLA', 'target', 'mutant']).agg(mean_pred=('pred', 'mean')).reset_index()
    test_mean_metrics = pd.DataFrame(get_metrics(test_mean_preds[tcol], test_mean_preds['mean_pred']), index=[0])
    test_mean_metrics.columns = [f'test_mean_{x}' for x in test_mean_metrics.columns]
    # Concat test:
    test_concat_metrics = pd.DataFrame(get_metrics(test_preds[tcol], test_preds[pcol]), index=[0])
    test_concat_metrics.columns = [f'test_concat_{x}' for x in test_concat_metrics.columns]
    # Concat Valid:
    total_valid_metrics = pd.DataFrame(get_metrics(valid_preds[tcol], valid_preds[pcol]), index=[0])
    total_valid_metrics.columns = [f'total_valid_{x}' for x in total_valid_metrics.columns]
    return pd.concat(
        [total_valid_metrics, test_mean_metrics, test_concat_metrics, pd.DataFrame(hyperparams, index=[0])], axis=1)


def main():
    print('Starting script')
    start = dt.now()
    # I like dictionary for args :-)
    args = vars(args_parser())
    maindir = args['dir'] + '/' if not args['dir'].endswith('/') else args['dir']
    # This is also the main filename for all subsequent subdirectories/file
    # Use this to recover the kwargs / params, or use the kwargs that are written to args.txt?
    fold_dirs = [x for x in os.listdir(args['dir']) if
                 path.isdir(os.path.join(maindir, x))]  # This should give all the hyperparams (200K+) combi
    wrapper = partial(pipeline, args=args)
    print('Doing fold dirs now')
    results = Parallel(n_jobs=args['n_cores'])(delayed(wrapper)(fold_dir=fd) for fd in tqdm(fold_dirs))
    pd.concat(results).to_csv(f'{maindir}total_results.csv', index=False)
    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f'Time elapsed: {elapsed[0]} minutes, {elapsed[1]} seconds.')


if __name__ == '__main__':
    main()
