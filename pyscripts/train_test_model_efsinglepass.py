import pandas as pd
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
from src.utils import str2bool, pkl_dump, mkdirs, get_random_id, get_datetime_string, plot_loss_aucs, \
    get_class_initcode_keys, make_filename, save_json
from src.torch_utils import save_checkpoint, load_checkpoint, save_model_full, get_available_device
from src.models import NNAlignEFSinglePass, NNAlignEFTwoStage
from src.train_eval import train_model_step, eval_model_step, predict_model, train_eval_loops
from sklearn.model_selection import train_test_split
from src.datasets import NNAlignDatasetEFSinglePass, NNAlignDataset
from src.data_processing import parse_fasta, load_structural_data
import numpy as np
from matplotlib import pyplot as plt
import tracemalloc
import seaborn as sns

import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a NNAlign model ')
    """
    Data processing args
    """
    parser.add_argument('-cuda', dest='cuda', required=False, type=str2bool, default=False,
                        help='Whether to activate Cuda. If true, will check if any gpu is available.')
    parser.add_argument('-trf', '--train_file', dest='train_file', required=True, type=str,
                        default='../data/aligned_icore/230530_cedar_aligned.csv',
                        help='filename of the train input file')
    parser.add_argument('-tef', '--test_file', dest='test_file', required=True, type=str,
                        default='../data/aligned_icore/230530_prime_aligned.csv',
                        help='filename of the test input file')
    parser.add_argument('-struc', '--structure_file', dest='structure_file', required=False, type=str,
                        help='Path to the structure file')
    parser.add_argument('-fasta', '--fasta_file', dest='fasta_file', required=False, type=str,
                        help='Path to the FASTA file')
    parser.add_argument('-o', '--out', dest='out', required=False,
                        type=str, default='', help='Additional output name')
    parser.add_argument('-tts', '--split', dest='split', required=False, type=int,
                        default=5,
                        help='Train Test Split ; How to split the train/test data (test size=1/X) if kf is None')
    # TODO: Carlos: here, use None for kf because the data is already split. I'll let you figure out how to call the columns
    #       and what to use with -x, -y, -max_len, etc.
    parser.add_argument('-kf', '--fold', dest='fold', required=False, type=int, default=None,
                        help='If added, will split the input file into the train/valid for kcv')
    parser.add_argument('-x', '--seq_col', dest='seq_col', default='sequence', type=str, required=False,
                        help='Name of the column containing sequences (inputs)')
    parser.add_argument('-y', '--target_col', dest='target_col', default='target', type=str, required=False,
                        help='Name of the column containing sequences (inputs)')
    parser.add_argument('-enc', '--encoding', dest='encoding', type=str, default='BL50LO', required=False,
                        help='Which encoding to use: onehot, BL50L0, BL62LO, BL62FREQ (default = BL50LO)')
    parser.add_argument('-ml', '--max_len', dest='max_len', type=int, required=True,
                        help='Maximum sequence length admitted ;' \
                             'Sequences longer than max_len will be removed from the datasets')
    parser.add_argument('-pad', '--pad_scale', dest='pad_scale', type=float, default=None, required=False,
                        help='Number with which to pad the inputs if needed; ' \
                             'Default behaviour is 0 if onehot, -12 is BLOSUM')
    parser.add_argument('-fc', '--feature_cols', dest='feature_cols', nargs='+', required=False,
                        help='Name of columns (str) to use as extra features, space separated.' \
                             'For example, to add 2 features Rank and Similarity, do: -ef Rank Similarity')
    parser.add_argument('-add_ps', '--add_pseudo_sequence', dest='add_pseudo_sequence', type=str2bool, default=False,
                        help='Whether to add pseudo sequence to the model (true/false)')
    # parser.add_argument('-ps', '--pseudo_seq_col', dest='pseudo_seq_col', default='pseudoseq', type=str, required=False,
    #                     help='Name of the column containing the MHC pseudo-sequences')
    parser.add_argument('-add_pfr', '--add_pfr', dest='add_pfr', type=str2bool, default=False,
                        help='Whether to add fixed-size (3) mean peptide flanking regions to the model (true/false)')
    parser.add_argument('-add_fr_len', '--add_fr_len', dest='add_fr_len', type=str2bool, default=False,
                        help='Whether to add length of the flanking regions of each motif to the model (true/false)')
    parser.add_argument('-add_pep_len', '--add_pep_len', dest='add_pep_len', type=str2bool, default=False,
                        help='Whether to add the peptide length encodings (as one-hot) to the model (true/false)')
    parser.add_argument('-min_clip', '--min_clip', dest='min_clip', type=int, default=None,
                        help='Whether to add the peptide length encodings (as one-hot) to the model (true/false)')
    parser.add_argument('-max_clip', '--max_clip', dest='max_clip', type=int, default=None,
                        help='Whether to add the peptide length encodings (as one-hot) to the model (true/false)')
    parser.add_argument('-indel', '--indel', dest='indel', type=str2bool, default=False,
                        help='Whether to add insertions/deletions')
    # TODO: Deprecate on_the_fly and set it as default behaviour in datasets (remove old behaviour)
    parser.add_argument('-otf', '--on_the_fly', dest='on_the_fly', type=str2bool, default=True,
                        help='Do MHC expansion on the fly vs saving everything in memory.'
                             'Now True by default, to be deprecated ')
    parser.add_argument('-add_str', '--add_structure', dest='add_structure', type=str2bool, default=False,
                        help='Whether to add structural data to the model (true/false)')
    parser.add_argument('-add_mean_str', '--add_mean_structure', dest='add_mean_structure', type=str2bool,
                        default=False,
                        help='Whether to add mean structural data to the model (true/false)')
    parser.add_argument('-two_stage', '--two_stage', dest='two_stage', type=str2bool, default=False,
                        help='Use 2stage model (for add_mean_structure)')
    parser.add_argument('-scols', '--struct_cols', dest='struct_cols', nargs='+',
                        default=['rsa', 'pq3_H', 'pq3_E', 'pq3_C', 'disorder'],
                        help='List of columns to include in the structural features. ')
    """
    Neural Net & Encoding args 
    """
    parser.add_argument('-nh', '--n_hidden', dest='n_hidden', required=True,
                        type=int, help='Number of hidden units')
    parser.add_argument('-std', '--standardize', dest='standardize', type=str2bool, default=False,required=False,
                        help='Whether to include standardization (True/False)')
    parser.add_argument('-bn', '--batchnorm', dest='batchnorm', type=str2bool, default=False, required=False,
                        help='Whether to add BatchNorm to the model (True/False)')
    parser.add_argument('-do', '--dropout', dest='dropout', type=float, default=0.0, required=False,
                        help='Whether to add DropOut to the model (p in float e[0,1], default = 0.0)')
    parser.add_argument('-ws', '--window_size', dest='window_size', type=int, default=9, required=False,
                        help='Window size for sub-mers selection (default = 6)')
    parser.add_argument('-efbn', '--batchnorm_ef', dest='batchnorm_ef',
                        default=False, type=str2bool,
                        help='Whether to add BatchNorm to the EF layer, (default = False)')
    parser.add_argument('-efdo', '--dropout_ef', dest='dropout_ef',
                        default=0.0, type=float,
                        help='Whether to add DropOut to the EF layer (p in float e[0,1], default = 0.0)')
    parser.add_argument('-add_hl', '--add_hidden_layer', dest='add_hidden_layer', type=str2bool, required=False,
                        default=False, help='Whether to add a second hidden layer (True/False)')
    parser.add_argument('-nh2', '--n_hidden_2', dest='n_hidden_2', required=False, default=10,
                        type=int, help='Number of hidden units for the additional second hidden layer (default = 10)')
    """
    Training hyperparameters & args
    """
    parser.add_argument('-br', '--burn_in', dest='burn_in', required=False, type=int, default=None,
                        help='Burn-in period (in int) to align motifs to P0. Disabled by default')
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=1e-4, required=False,
                        help='Learning rate for the optimizer')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', type=float, default=1e-4, required=False,
                        help='Weight decay for the optimizer')  # try 1e-3, 1e-4, 1e-6
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=128, required=False,
                        help='Batch size for mini-batch optimization')  # try 32, 64, 256
    parser.add_argument('-ne', '--n_epochs', dest='n_epochs', type=int, default=500, required=False,
                        help='Number of epochs to train')
    parser.add_argument('-tol', '--tolerance', dest='tolerance', type=float, default=1e-5, required=False,
                        help='Tolerance for loss variation to log best model')
    parser.add_argument('-rid', '--random_id', dest='random_id', type=str, default=None,
                        help='Adding a random ID taken from a batchscript that will start all crossvalidation folds. Default = ""')
    parser.add_argument('-debug', dest='debug', type=str2bool, default=False,
                        help='Turning on debug mode')
    return parser.parse_args()


"""
Using this script now as a way to run the train and test in a single file because it is easier to deal with due to the random
unique ID and k-fold crossvalidation process. I could rewrite some bashscript to move all the resulting folders somewhere, 
then ls that somewhere and iterate through each of the folders to reload each model & run individually in each script, but here 
we can do this instead.
"""


def main():
    start = dt.now()
    tracemalloc.start()
    # I like dictionary for args :-)
    args = vars(args_parser())

    if args['add_mean_structure'] and args['add_structure']:
        raise ValueError("--add_mean_structure and --add_structrue can't both be True. Only one can be active at a time")
    if args['add_mean_structure'] and len(args['struct_cols']) != 5:
        raise ValueError(f"--add_mean_structure is set as True but using {len(args['struct_cols'])} structure columns. 5 columns are required for now (To be changed)")
    if (args['add_mean_structure'] and not args['two_stage']) or (not args['add_mean_structure'] and args['two_stage']):
        raise ValueError(
            f'add_mean_structure, two-stage model must both be active! Currently: {args["add_mean_structure"], args["two_stage"]};\n(set --add_mean_structure True --two_stage True instead!)')
    # Cuda activation
    if torch.cuda.is_available() and args['cuda']:
        device = get_available_device()
    else:
        device = torch.device('cpu')
    print("Using : {}".format(device))
    if args['min_clip'] is not None and args['max_clip'] is not None and args['add_pep_len']:
        assert args['min_clip'] < args[
            'max_clip'], "args['min_clip'] should be smaller than args['max_clip'] for adding pep lens" \
                         f"Got (min, max) = ({args['min_clip'], args['max_clip']} instead"
    # File-saving stuff
    unique_filename, kf, rid, connector = make_filename(args)

    checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
    outdir = os.path.join('../output/', unique_filename) + '/'
    mkdirs(outdir)
    df = pd.read_csv(args['train_file'])
    if args['debug']:
        df = df.sample(min(5000, len(df)), random_state=13)
    tmp = args['seq_col']

    # Filtering from training set
    test_df = pd.read_csv(args['test_file'])

    if args['fold'] is not None:
        torch.manual_seed(args['fold'])
        fold = args['fold']
        dfname = os.path.basename(args['train_file']).split('.')[0]
        train_df = df.query('fold!=@fold')
        valid_df = df.query('fold==@fold')
        unique_filename = f'kcv_{dfname}_f{fold:02}_{unique_filename}'
        checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
    else:
        torch.manual_seed(0)
        np.random.seed(0)
        train_df, valid_df = train_test_split(df, test_size=1 / args["split"])

    # TODO : what to do with this ?
    # Quick hotfix because i don't know why this query/eval thing suddenly changed and stopped working
    # tmpvals = train_df[tmp].values
    # This
    # test_df = test_df.query(f'{tmp} not in @train_df.{tmp}.values')

    MODELCLASS = NNAlignEFTwoStage if args['two_stage'] else NNAlignEFSinglePass
    DATASETCLASS = NNAlignDataset if args['on_the_fly'] else NNAlignDatasetEFSinglePass
    # Def params so it's ✨tidy✨, using get_class_initcode to get the keys needed to init a class
    model_keys = get_class_initcode_keys(MODELCLASS, args)
    # Here UglyWorkAround exist to give the __init__ code to dataset because I'm currently using @profile
    dataset_keys = get_class_initcode_keys(DATASETCLASS, args)
    args['on_the_fly'] = True
    model_params = {k: args[k] for k in model_keys}
    dataset_params = {k: args[k] for k in dataset_keys}
    optim_params = {'lr': args['lr'], 'weight_decay': args['weight_decay']}
    # if args['add_structure']:
    #     structural_data = load_structural_data(args['structure_file'])
    #     fasta_data = parse_fasta(args['fasta_file'])
    #     dataset_params['structural_data'] = structural_data
    #     dataset_params['fasta_data'] = fasta_data
    # Define dimensions for extra features added
    model_params['pseudoseq_dim'] = 680 if args['add_pseudo_sequence'] else 0
    model_params['feat_dim'] = 0
    model_params['matrix_dim'] = 20 + len(args['struct_cols']) if args['add_structure'] else 20
    if args['add_pfr']:
        model_params['feat_dim'] += 40
    if args['add_fr_len']:
        model_params['feat_dim'] += 4
    if args['add_pep_len']:
        max_clip = args['max_clip'] if args['max_clip'] is not None else args['max_len']
        min_clip = args['min_clip'] if args['min_clip'] is not None else df[args['seq_col']].apply(len).min()
        model_params['feat_dim'] += max_clip - min_clip + 2

    # TODO : Here, this shouldn't be enabled because the structure features are directly concatenated to the encoded vectors
    #        and model.feat_dim is for the extra features that are concatenated to the end of the flattened vector
    # if args['add_structure']:
    #     model_params['feat_dim'] += 5 #

    model_params['feat_dim'] = int(model_params['feat_dim'])
    model = MODELCLASS(activation=nn.ReLU(), **model_params)
    model.to(device)
    # Here changed the loss to MSE to train with sigmoid'd output values instead of labels
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), **optim_params)
    # if args['on_the_fly']:
    # TODO Quick workaround
    train_dataset = DATASETCLASS(train_df, **dataset_params)
    valid_dataset = DATASETCLASS(valid_df, **dataset_params)
    test_dataset = DATASETCLASS(test_df, **dataset_params)
    _, dataset_peak = tracemalloc.get_traced_memory()
    #
    # else:
    #     train_dataset = NNAlignDatasetEFSinglePass(train_df, **dataset_params)
    #     valid_dataset = NNAlignDatasetEFSinglePass(valid_df, **dataset_params)
    #     test_dataset = NNAlignDatasetEFSinglePass(test_df, **dataset_params)
    #     _, dataset_peak = tracemalloc.get_traced_memory()

    train_loader = train_dataset.get_dataloader(batch_size=args['batch_size'], sampler=RandomSampler)
    valid_loader = valid_dataset.get_dataloader(batch_size=args['batch_size'] * 2, sampler=SequentialSampler)
    test_loader = test_dataset.get_dataloader(batch_size=args['batch_size'] * 2, sampler=SequentialSampler)
    # Training loop & train/valid results

    model, train_metrics, valid_metrics, train_losses, valid_losses, \
    best_epoch, best_val_loss, best_val_auc = train_eval_loops(args['n_epochs'], args['tolerance'], model, criterion,
                                                               optimizer,
                                                               train_dataset, train_loader, valid_loader,
                                                               checkpoint_filename,
                                                               outdir, args['burn_in'], args['standardize'])
    _, traineval_peak = tracemalloc.get_traced_memory()

    pkl_dump(train_losses, f'{outdir}/train_losses_{unique_filename}.pkl')
    pkl_dump(valid_losses, f'{outdir}/valid_losses_{unique_filename}.pkl')
    pkl_dump(train_metrics, f'{outdir}/train_metrics_{unique_filename}.pkl')
    pkl_dump(valid_metrics, f'{outdir}/valid_metrics_{unique_filename}.pkl')
    train_aucs = [x['auc'] for x in train_metrics]
    valid_aucs = [x['auc'] for x in valid_metrics]
    plot_loss_aucs(train_losses, valid_losses, train_aucs, valid_aucs,
                   unique_filename, outdir, 150)

    # Reload the model and predict
    print('Reloading best model and returning validation and test predictions')
    model = load_checkpoint(model, checkpoint_filename, outdir)

    # validation set
    valid_preds = predict_model(model, valid_dataset, valid_loader)
    print('Saving valid predictions from best model')
    valid_preds.to_csv(f'{outdir}valid_predictions_{unique_filename}.csv', index=False)
    # Test set
    test_preds = predict_model(model, test_dataset, test_loader)
    # test_loss, test_metrics = eval_model_step(model, criterion, test_loader)
    print('Saving test predictions from best model')
    test_fn = os.path.basename(args['test_file']).split('.')[0]
    test_preds.to_csv(f'{outdir}test_predictions_{test_fn}_{unique_filename}.csv', index=False)
    tracemalloc.stop()

    end = dt.now()
    elapsed = divmod((end - start).seconds, 60)
    # Saving text file for the run:
    with open(f'{outdir}args_{unique_filename}.txt', 'w') as file:
        header = "#" * 100 + "\n#" + " " * 42 + "PARAMETERS" + "\n" + '#' * 100 + '\n'
        file.write(header)
        for key, value in args.items():
            file.write(f"{key}: {value}\n")
        header2 = "#" * 100 + "\n#" + " " * 42 + "VALID-TEST\n" + '#' * 100 + '\n'
        file.write(header2)
        file.write(f"Best valid epoch: {best_epoch}\n")
        file.write(f"Best valid loss: {best_val_loss}\n")
        file.write(f"Best valid auc: {best_val_auc}\n")
        file.write(f"Test file: {args['test_file']}\n")
        # file.write(f"Test loss: {test_loss}\n")
        # file.write(f"Test AUC: {test_metrics['auc']}\n")
        file.write(f"Elapsed time: {elapsed[0]} minutes {elapsed[1]} seconds.")
    save_model_full(model, checkpoint_filename, outdir, dict_kwargs=model_params)
    save_json(args, f'run_parameters_{unique_filename}.json', outdir)

    print(f"dataset_peak memory usage: {dataset_peak / (1024 ** 2):.2f} MB")
    print(f"traineval_peak memory usage: {traineval_peak / (1024 ** 2):.2f} MB")
    print(f'Program finished in {elapsed[0]} minutes, {elapsed[1]} seconds.')
    sys.exit(0)


if __name__ == '__main__':
    main()
