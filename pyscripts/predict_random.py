import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch import optim
from torch import nn
from torch.utils.data import SequentialSampler, RandomSampler
from datetime import datetime as dt
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_loss_aucs
from src.torch_utils import save_checkpoint, load_checkpoint
from src.models import NNAlignEFSinglePass
from src.train_eval import train_model_step, eval_model_step, predict_model, train_eval_loops
from sklearn.model_selection import train_test_split
from src.datasets import get_NNAlign_dataloaderEFSinglePass
from src.data_processing import encode_batch, encode_batch_weighted, PFR_calculation, FR_lengths, pep_len_1hot
from matplotlib import pyplot as plt
import seaborn as sns

import argparse

# Function to treat boolean entries
def str2bool(v):
    """Converts str to bool from argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args_parser():
    parser = argparse.ArgumentParser(description='Script to predict random peptides from the trained models')
    """
    Data processing args
    """
    parser.add_argument('-tef', '--test_file', dest='test_file', required=True, type=str,
                        default='../data/carlos/15mer_ranpep_DRB1_0301.txt',
                        help='Filename of the test input file (random peptides), with the correct structure data frame')
    parser.add_argument('-dir', '--model_dir', dest='model_dir', required=True, type=str,
                        default='../output/231116_complete_wd1e-4/',
                        help='Directory with the saved best PyNNAlign models (in case of cross-validation)')
    parser.add_argument('-od', '--outdir', dest='outdir', required=True, type=str,
                        default='../testpred/231116_complete_wd1e-4_ranpeps/',
                        help='Directory to store the test predictions')
    parser.add_argument('-of', '--outfile', dest='outfile', required=True, type=str,
                        default='15mer_ranpep_pred',
                        help='Filename of the test predictions')
    parser.add_argument('-nh', '--n_hidden', dest='n_hidden', required=False, default=60,
                        type=int, help='Number of hidden units, default = 60')
    parser.add_argument('-std', '--standardize', dest='standardize', type=str2bool, required=False, default=False,
                        help='Whether to include standardization (True/False), default = False')
    parser.add_argument('-bn', '--batchnorm', dest='batchnorm', type=str2bool, required=False, default=False,
                        help='Whether to add BatchNorm to the model (True/False), default = False')
    parser.add_argument('-do', '--dropout', dest='dropout', type=float, default=0.0, required=False,
                        help='Whether to add DropOut to the model (p in float e[0,1], default = 0.0)')
    parser.add_argument('-ws', '--window_size', dest='window_size', type=int, default=9, required=False,
                        help='Window size for sub-mers selection (default = 9)')
    parser.add_argument('-ef', '--ef_dim', dest='ef_dim', type=int, default=734, required=False,
                        help='Extra-feature dimension (default = 734)')
    return parser.parse_args()

# Parse the command-line arguments and store them in the 'args' variable
args = args_parser()
args_dict = vars(args)

# Keeping a list of all directories with each model
model_files = []
for root, dirs, files in os.walk(args_dict['model_dir']):
    for file in files:
        if file.startswith("checkpoint_best"):
            file_path = os.path.join(root, file)
            # print(file_path)
            model_files.append(file_path)

# Read the test data file and load it according to the PyNNAlign scripts
test_data = pd.read_csv(args_dict['test_file'])
test_loader, test_dataset = get_NNAlign_dataloaderEFSinglePass(test_data, indel=False, sampler=SequentialSampler,
                                                               return_dataset=True, max_len=15, window_size=9, encoding='BL50LO',
                                                               seq_col='Sequence', target_col='BA', pad_scale=None, 
                                                               batch_size=128, add_pseudo_sequence=True, pseudo_seq_col='pseudoseq',
                                                               add_pfr=True, add_fr_len=True, add_pep_len=True)

# Define the model parameters
model = NNAlignEFSinglePass(activation = nn.ReLU(), extrafeat_dim = args_dict['ef_dim'], indel = False, 
                            n_hidden = args_dict['n_hidden'], window_size = args_dict['window_size'], 
                            batchnorm = args_dict['batchnorm'], dropout = args_dict['dropout'], 
                            standardize = args_dict['standardize'])

# Make predictions using each model for each sample
for i in range(0, len(model_files), 1):
    
    # Reload the best model
    checkpoint_filename = model_files[i]
    model = load_checkpoint(model, checkpoint_filename)
    
    # Model predictions
    test_preds = predict_model(model, test_dataset, test_loader)

    print(f'Saving test predictions from the best model of fold {i+1}')

    # Use os.path.basename to get the last component of the path
    test_out = args_dict['outdir'] + args_dict['outfile'] + '_fold' + str(i+1) + '.csv'
    test_preds.to_csv(test_out, index=False)