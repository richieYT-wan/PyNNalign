import pandas as pd
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import glob
import argparse
from joblib import Parallel, delayed
from functools import partial
from src.utils import mkdirs, str2bool

def args_parser():
    parser = argparse.ArgumentParser(description='Script to get a set of predicted motifs from a PyNNalign output'\
        'Assumes that the input directory contains either all the crossvalidation directories or a single prediction file')
    """
    Data processing args
    """
    parser.add_argument('-i', '--indir', dest='indir', type=str, 
        help='Directory containing either the 5 fold crossvalidation directories, or a single validation prediction file, use the flag -kf to True if it contains directories, or False if it contains as single file')
    parser.add_argument('-o', '--outdir', dest='outdir', type=str, default=None,
        help='Directory in which to save the outputs. If None, then will save in the input folder')
    parser.add_argument('-fn', '--filename', dest='filename', type=str, default='',
        help='Additional filename identifier')
    parser.add_argument('-l', '--len', dest='len', type=int, default=9, nargs='+',
        help='Length to filter by, will also add it to the filename.'\
        'Can one or more lengths, space separated. Example: -l 8 ; or -l 8 9 10')
    parser.add_argument('-kf', dest='kf', type=str2bool, default=True,
        help='Whether the input directory contains other KF directories or a single file. True==contains directories, False==read a single file')
    parser.add_argument('-hla', dest='hla', nargs='+', type=str, default='HLA-A0201',
        help='One or more HLA allele to filter by, space separated. Example: -hla HLA-A0201 ; or -hla HLA-A0201 HLA-A0101')
    return parser.parse_args()


def wrapper(dfs, concat_df, hla, length, args, outdir, unique_filename):
    fn = f'{hla}_l{length:02}_{unique_filename}'.replace(':','')
    if length==8 and 'indel_False' in args['indir']:return 0
    if args['kf']:
        for i,df in enumerate(dfs):
            df.query('HLA==@hla and len==@length')['motif'].to_csv(f'{outdir}/{fn}_kcv_{i}.txt', index=False, header=False)
        concat_df.query('HLA==@hla and len==@length')['motif'].to_csv(f'{outdir}/{fn}_allpartitions_concat.txt', index=False, header=False)
    else:
        dfs[0].query('HLA==@hla and len==@length')['motif'].to_csv(f'{outdir}/{fn}.txt', index=False, header=False)



def main():
    args = vars(args_parser())
    hlas_list = args['hla']
    if type(hlas_list)!=list:
        hlas_list = list(hlas_list)
    len_list = args['len']
    if type(len_list) != list:
        len_list = list(len_list)

    unique_filename = args['filename']
    indir = args['indir']
    outdir = indir if args['outdir'] is None else args['outdir']
    mkdirs(outdir)

    input_files = glob.glob(f'{indir}/*/*valid_predictions*.csv') if args['kf'] else glob.glob(f'{indir}*valid_predictions*.csv') 
    input_files = sorted(input_files)
    print(indir)
    print('\n\n', input_files, '\n\n')
    dfs = [pd.read_csv(x).query('target==1') for x in input_files]
    if args['kf']:
        dfs = [df.assign(k=i) for i,df in enumerate(dfs)]
        concat_df = pd.concat(dfs)
    else:
        concat_df = None

    if 'indel_False' in args['indir'] and 8 in len_list:
        len_list.pop(len_list.index(8))

    for length in len_list:
        wr = partial(wrapper, dfs=dfs, concat_df=concat_df, length = length, args=args, outdir=outdir, unique_filename=unique_filename)
        Parallel(n_jobs=8)(delayed(wr)(hla=hla) for hla in hlas_list)


if __name__ == '__main__':
    main()





def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a VAE model with all chains')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', default=False, type=str2bool,
                        help="Will use GPU if True and GPUs are available")
    parser.add_argument('-logwb', '--log_wandb', dest='log_wandb', required=False, default=False,
                        type=str2bool, help='Whether to log a run using WandB. False by default')
    parser.add_argument('-f', '--file', dest='file', required=False, type=str,
                        default='../data/filtered/231205_nettcr_old_26pep_with_swaps.csv',
                        help='filename of the input train file')
    parser.add_argument('-tf', '--test_file', dest='test_file', type=str,
                        default=None, help='External test set (None by default)')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default='', help='Additional output name')
    parser.add_argument('-a1', '--a1_col', dest='a1_col', default='A1', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-a2', '--a2_col', dest='a2_col', default='A2', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-a3', '--a3_col', dest='a3_col', default='A3', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b1', '--b1_col', dest='b1_col', default='B1', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b2', '--b2_col', dest='b2_col', default='B2', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')
    parser.add_argument('-b3', '--b3_col', dest='b3_col', default='B3', type=str, required=False,
                        help='Name of the column containing B3 sequences (inputs)')

    parser.add_argument('-mla1', '--max_len_a1', dest='max_len_a1', type=int, default=7,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla2', '--max_len_a2', dest='max_len_a2', type=int, default=8,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mla3', '--max_len_a3', dest='max_len_a3', type=int, default=22,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb1', '--max_len_b1', dest='max_len_b1', type=int, default=6,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb2', '--max_len_b2', dest='max_len_b2', type=int, default=7,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlb3', '--max_len_b3', dest='max_len_b3', type=int, default=23,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-mlpep', '--max_len_pep', dest='max_len_pep', type=int, default=12,
                        help='Max seq length admitted for peptide. Set to 0 to disable adding peptide to the input')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50LO, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-pad', '--pad_scale', dest='pad_scale', type=float, default=None, required=False,
                        help='Number with which to pad the inputs if needed; ' \
                             'Default behaviour is 0 if onehot, -20 is BLOSUM')
    parser.add_argument('-addpe', '--add_positional_encoding', dest='add_positional_encoding', type=str2bool, default=False,
                        help='Adding positional encoding to the sequence vector. False by default')
    """
    Models args 
    """
    parser.add_argument('-nhtcr', '--hidden_dim_tcr', dest='hidden_dim_tcr', type=int, default=200,
                        help='Number of hidden units in the VAE. Default = 200')
    parser.add_argument('-nhpep', '--hidden_dim_pep', dest='hidden_dim_pep', type=int, default=200,
                        help='Number of hidden units in the VAE. Default = 200')
    parser.add_argument('-nl', '--latent_dim', dest='latent_dim', type=int, default=100,
                        help='Size of the latent dimension. Default = 100')
    parser.add_argument('-act', '--activation', dest='activation', type=str, default='selu',
                        help='Which activation to use. Will map the correct nn.Module for the following keys:' \
                             '[selu, relu, leakyrelu, elu]')
    # Not implemented as of now
    parser.add_argument('-do', dest='dropout', type=float, default=0.25,
                        help='Dropout percentage in the hidden layers (0. to disable)')
    parser.add_argument('-bn', dest='batchnorm', type=str2bool, default=True,
                        help='Use batchnorm (True/False)')
    """
    Training hyperparameters & args
    """
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=5e-4, required=False,
                        help='Learning rate for the optimizer. Default = 5e-4')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', type=float, default=1e-4, required=False,
                        help='Weight decay for the optimizer. Default = 1e-4')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=512, required=False,
                        help='Batch size for mini-batch optimization')
    parser.add_argument('-ne', '--n_epochs', dest='n_epochs', type=int, default=5000, required=False,
                        help='Number of epochs to train')
    parser.add_argument('-tol', '--tolerance', dest='tolerance', type=float, default=1e-5, required=False,
                        help='Tolerance for loss variation to log best model')
    parser.add_argument('-lwseq', '--weight_seq', dest='weight_seq', type=float, default=1,
                        help='Which beta to use for the seq reconstruction term in the loss')
    parser.add_argument('-lwkld_n', '--weight_kld_n', dest='weight_kld_n', type=float, default=1e-2,
                        help='Which weight to use for the KLD (normal) term in the loss')
    parser.add_argument('-lwkld_z', '--weight_kld_z', dest='weight_kld_z', type=float, default=1,
                        help='Which weight to use for the KLD (Latent) term in the loss')
    parser.add_argument('-wukld', '--warm_up_kld', dest='warm_up_kld', type=int, default=10,
                        help='Whether to do a warm-up period for the loss (without the KLD term). ' \
                             'Default = 10. Set to 0 if you want this disabled')
    parser.add_argument('-kldts', '--kld_tahn_scale', dest='kld_tahn_scale', type=float, default=0.1,
                        help='Scale for the TanH annealing in the KLD_n term')
    parser.add_argument('-fp', '--flat_phase', dest='flat_phase', default=None, type=int,
                        help='If used, the duration (in epochs) of the "flat phase" in the KLD annealing')

    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Whether to run in debug mode (False by default)')
    parser.add_argument('-pepweight', dest='pep_weighted', type=str2bool, default=False,
                        help='Using per-sample (by peptide label) weighted loss')
    # TODO: TBD what to do with these!
    """
    TODO: Misc. 
    """
    
    parser.add_argument('-kf', '--fold', dest='fold', required=False, type=int, default=None,
                        help='If added, will split the input file into the train/valid for kcv')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-seed', '--seed', dest='seed', type=int, default=None,
                        help='Torch manual seed. Default = 13')
    return parser.parse_args()
