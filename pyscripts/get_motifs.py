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
    fn = f'{hla}_length_{length}_{unique_filename}'.replace(':','')
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
    print('\n\n', input_files, '\n\n')
    dfs = [pd.read_csv(x) for x in input_files]

    if args['kf']:
        dfs = [df.assign(k=i) for i,df in enumerate(dfs)]
        concat_df = pd.concat(dfs)
    else:
        concat_df = None

    for length in len_list:
        wr = partial(wrapper, dfs=dfs, concat_df=concat_df, length = length, args=args, outdir=outdir, unique_filename=unique_filename)
        Parallel(n_jobs=8)(delayed(wr)(hla=hla) for hla in hlas_list)

            

if __name__ == '__main__':
    main()
