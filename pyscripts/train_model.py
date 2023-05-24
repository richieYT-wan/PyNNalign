import pandas as pd
from tqdm.auto import tqdm
import os, sys
import torch
from torch import optim
from torch import nn
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string
from src.torch_utils import save_checkpoint, load_checkpoint
from src.models import NNAlign
from src.train_eval import train_model_step, eval_model_step, predict_model
from sklearn.model_selection import train_test_split
from src.datasets import get_NNAlign_dataloader

import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a NNAlign model ')
    #
    # parser.add_argument('-tf', '--train_file', dest='train', required=True,
    #                     type=str, help='filename of the train_file, w/ extension & full path' \
    #                                    'ex: /path/to/file/train.csv')
    # parser.add_argument('-vf', '--valid_file', dest='valid', required=True,
    #                     type=str, help='filename of the valid_file, w/ extension & full path' \
    #                                    'ex: /path/to/file/valid.csv')

    """
    Data processing args
    """
    parser.add_argument('-f', '--file', dest='file', required=True, type=str,
                        default='../data/NetMHCIIpan_train/drb1_0301.csv',
                        help='filename of the input file')
    parser.add_argument('-s', '--split', dest='split', required=False, type=int,
                        default=5, help=('How to split the train/test data (test size=1/X)'))
    parser.add_argument('-k', '--fold', dest='fold', required=False, type=int, default=None,
                        help='If added, will split the input file into the train/valid for kcv')
    parser.add_argument('-x', '--seq_col', dest='seq_col', default='Sequence', type=str, required=True,
                        help='Name of the column containing sequences (inputs)')
    parser.add_argument('-y', '--target_col', dest='target_col', default='target', type=str, required=False,
                        help='Name of the column containing sequences (inputs)')
    parser.add_argument('-o', '--outdir', dest='outdir', required=True,
                        type=str, default='../output/', help='Where to save results')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50L0, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-ml', '--max_len', dest='max_len', type=int, required=True,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-pad', '--pad_scale', dest='pad_scale', type=float, default=None, required=False,
                        help='Number with which to pad the inputs if needed; ' \
                             'Default behaviour is 0 if onehot, -12 is BLOSUM')
    """
    Neural Net & Encoding args 
    """
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
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=1e-4, required=False,
                        help='Learning rate for the optimizer')
    parser.add_argument('-wd', '--weight_decay', dest='wd', type=float, default=1e-2, required=False,
                        help='Weight decay for the optimizer')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=128, required=False,
                        help='Batch size for mini-batch optimization')
    parser.add_argument('-ne', '--n_epochs', dest='n_epochs', type=int, default=500, required=False,
                        help='Number of epochs to train')
    parser.add_argument('tol', '--tolerance', dest='tolerance', type=float, default=1e-5, required=False,
                        help='Tolerance for loss variation to log best model')
    return parser.parse_args()


def main():
    start = dt.now()
    # I like dictionary for args :-)
    args = vars(args_parser())
    # File-saving stuff
    unique_filename = f'{get_datetime_string()}_{get_random_id(4)}'
    outdir = os.path.join(args["outdir"], unique_filename)
    mkdirs(outdir)

    print(args)
    # TODO: Deprecate this behaviour for now because I don't want to deal with it
    # train_df = pd.read_csv(args['train'])
    # valid_df = pd.read_csv(args['valid'])
    df = pd.read_csv(args['file'])
    if args['fold'] is not None:
        train_df = df.query('fold!=@args["fold"]')
        valid_df = df.query('fold==@args["fold"]')
    else:
        train_df, valid_df = train_test_split(df, test_size=1 / args["split"])
    # TODO: For now we are doing like this because we don't care about other activations, singlepass, indels
    # Def params so it's ✨tidy✨
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
    train_metrics, valid_metrics = [], []
    best_val_loss = 100
    best_val_auc = 0
    best_epoch = 1

    print('Starting training cycles')
    for e in tqdm(range(1, args['n_epochs'] + 1), desc='epochs'):
        train_loss, train_metric = train_model_step(model, criterion, optimizer, train_loader)
        valid_loss, valid_metric = eval_model_step(model, criterion, valid_loader)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss<=best_val_loss+args['tolerance'] and valid_metric['auc']>best_val_auc:
            best_epoch = e
            best_val_loss = valid_loss
            best_val_auc = best_val_auc
            save_checkpoint(model, filename=f'checkpoint_epoch_{best_epoch}_{unique_filename}.pt', dir_path=outdir)

    print(f'End of training cycles')
    print(f'Best train loss:\t{min(train_losses):.3e}, best train AUC:\t{max([x["auc"] for x in train_metrics])}')
    print(f'Best valid epoch: {best_epoch}')
    print(f'Best valid loss :\t{best_val_loss:.3e}, best train AUC:\t{best_val_auc}')
    print('Reloading best model and returning validation predictions')
    model = load_checkpoint(model, filename=f'checkpoint_epoch_{best_epoch}_{unique_filename}.pt',
                            dir_path=outdir)
    valid_preds = predict_model(model, valid_dataset, args['batch_size'])
    print('Saving valid predictions from best model')
    valid_preds.to_csv(f'{outdir}valid_predictions_{unique_filename}.csv', index=False)
    pkl_dump(train_losses, f'{outdir}train_losses_{unique_filename}.pkl')
    pkl_dump(valid_losses, f'{outdir}valid_losses_{unique_filename}.pkl')
    pkl_dump(train_metrics, f'{outdir}train_metrics_{unique_filename}.pkl')
    pkl_dump(valid_metrics, f'{outdir}valid_metrics_{unique_filename}.pkl')
    # Saving arguments for the run:
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")
    end = dt.now()
    elapsed = divmod((end-start).seconds, 60)
    print(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')
    sys.exit(0)

if __name__ == '__main__':
    main()
