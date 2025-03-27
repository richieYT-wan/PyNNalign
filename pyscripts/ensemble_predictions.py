import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os, sys
import torch
from torch import optim
from torch import nn
from torch.utils.data import SequentialSampler, RandomSampler
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tracemalloc
import seaborn as sns
import glob
import argparse
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_loss_aucs, \
    get_class_initcode_keys, make_filename, find_args
from src.torch_utils import load_model_full, load_checkpoint, save_model_full, get_available_device
from src.models import NNAlignEFSinglePass, NNAlignEFTwoStage
from src.train_eval import train_model_step, eval_model_step, predict_model, train_eval_loops
from src.datasets import NNAlignDataset


def args_parser():
    parser = argparse.ArgumentParser(description='Script to load multiple models and do ensemble prediction; Assumes all models (.pt) are put in one folder containing the JSON to reload it.')
    """
    Data processing args
    """
    # Model loading
    parser.add_argument('-cuda', dest='cuda', required=False, type=str2bool, default=False,
                        help='Whether to activate Cuda. If true, will check if any gpu is available.')
    """
    Models args 
    """
    parser.add_argument('-model_folder', type=str, required=False, default=None,
                        help='Path to the folder containing both the checkpoint and json file. ' \
                             'Should contain the .pt for all 5 models and one json file')
    # parser.add_argument('-pt_file', type=str, required=False,
    #                     default=None, help='Path to the checkpoint file to reload the VAE model')
    # parser.add_argument('-json_file', type=str, required=False,
    #                     default=None, help='Path to the json file to reload the VAE model')
    # Data in/out
    parser.add_argument('-tef', '--test_file', dest='test_file', required=True, type=str,
                        default='../data/aligned_icore/230530_prime_aligned.csv',
                        help='filename of the test input file')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default='', help='Additional output name')
    parser.add_argument('-kf', '--fold', dest='fold', required=False, type=int, default=None,
                        help='If added, will split the input file into the train/valid for kcv')
    # Dataset parameters
    parser.add_argument('-x', '--seq_col', dest='seq_col', default='sequence', type=str, required=False,
                        help='Name of the column containing sequences (inputs)')
    parser.add_argument('-y', '--target_col', dest='target_col', default='target', type=str, required=False,
                        help='Name of the column containing sequences (inputs)')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50L0, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Turning on debug mode')
    return parser.parse_args()


def main():
    start = dt.now()
    tracemalloc.start()
    # I like dictionary for args :-)
    args = vars(args_parser())
    # Cuda activation
    if torch.cuda.is_available() and args['cuda']:
        device = get_available_device()
    else:
        device = torch.device('cpu')
    print("Using : {}".format(device))

    # File-saving stuff
    unique_filename, kf, rid, connector = make_filename(args)
    outdir = os.path.join('../output/', unique_filename) + '/'
    mkdirs(outdir)
    print('Reading df')
    test_df = pd.read_csv(args['test_file'])

    if not args['model_folder'].endswith('/'):args['model_folder']=args['model_folder']+'/'
    # reload params
    params = find_args(args['model_folder'])

    # Define dimensions for extra features added
    pseudoseq_dim = 680 if params['add_pseudo_sequence'] else 0
    feat_dim = 0
    if params['add_pfr']:
        feat_dim += 40
    if params['add_fr_len']:
        feat_dim += 4
    if params['add_pep_len']:
        max_clip = params['max_clip'] if params['max_clip'] is not None else params['max_len']
        min_clip = params['min_clip'] if params['min_clip'] is not None else test_df[params['seq_col']].apply(len).min()
        feat_dim += max_clip - min_clip + 2

    # TODO:  Hotfix:
    extra_dict = {'pseudoseq_dim': pseudoseq_dim, 'feat_dim': feat_dim}

    model_json = glob.glob(f'{args["model_folder"]}*JSON_kwargs*.json')[0]
    models_pts = sorted(glob.glob(f'{args["model_folder"]}*.pt'))
    assert len(models_pts)>0 and len(model_json)>0, f'No models found! JSON: {model_json}, .pts: {models_pts}'
    models = [load_model_full(pt_file, model_json, extra_dict=extra_dict, return_json=False).eval() for pt_file in models_pts]

    dataset_keys = get_class_initcode_keys(NNAlignDataset, params)
    dataset_params = {k: params[k] for k in dataset_keys}

    for model in models:
        model.to(device)
    # Here changed the loss to MSE to train with sigmoid'd output values instead of labels
    print('Getting dataset')
    test_dataset = NNAlignDataset(test_df, **dataset_params)
    _, dataset_peak = tracemalloc.get_traced_memory()
    test_loader = test_dataset.get_dataloader(batch_size=params['batch_size'] * 2, sampler=SequentialSampler)
    print('Running predictions')
    # Test set
    test_preds = pd.concat([predict_model(model, test_dataset, test_loader, verbose=True).assign(model_n=i) for i, model in enumerate(models)])
    # test_loss, test_metrics = eval_model_step(model, criterion, test_loader)
    print('Saving test predictions from best model')
    test_fn = os.path.basename(args['test_file']).split('.')[0]
    test_preds.to_csv(f'{outdir}test_predictions_CONCAT_{test_fn}_{unique_filename}.csv', index=False)
    tracemalloc.stop()

    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    print(f"dataset_peak memory usage: {dataset_peak / (1024 ** 2):.2f} MB")
    print(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')
    sys.exit(0)


if __name__ == '__main__':
    main()
