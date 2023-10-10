import pandas as pd
import numpy as np
from datetime import datetime
import os

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
    parser = argparse.ArgumentParser(description='Script to process the test results from the NNAlign algorithm')

    """
    Data arguments
    """
    parser.add_argument('-f', '--test_results', dest='test_results', required=True, type=str,
                        help='Filename of the test results file')
    parser.add_argument('-bn', '--keep_bn', dest='keep_bn', type=str2bool, required=False, default='f',
                        help='Whether to keep the basename of the input file (True/False)')
    parser.add_argument('-id', '--file_id', dest='file_id', required=False, default='Out', type=str, 
                        help='ID of the input file (used for recognizing the basename of your file). E.g.: if your basename is "Try1_251023_...", then you can use "Try1" as your ID)')
    parser.add_argument('-rv', '--rank_val', dest='rank', required=False, default=200, type=int, 
                        help='Value to define the list of the results with the highest predicted scores')
    parser.add_argument('-add_1mhc', '--add_1mhc', dest='add_1mhc', type=str2bool, required=False, default='f',
                        help='Whether to obtain a ranked-list of motifs for a concrete MHC (True/False)')
    parser.add_argument('-unique_mhc', '--mhc_unique', dest='mhc_unique', required=False, default='DRB1_0101', type=str,
                        help='MHC name for obtaining a specific ranked-list of motifs')
    parser.add_argument('-ba', '--ba_filter', dest='ba_filter', type=str2bool, required=False, default='True',
                        help='Whether to filter the results by binders (True/False)')
    parser.add_argument('-thr', '--ba_thr', dest='ba_thr', type=float, required=False, default=0.5,
                        help='Threshold value to exclude binders and not binders')
    parser.add_argument('-mhc_count', '--mhc_count', dest='mhc_count', type=int, required=True, default=1,
                        help='Threshold value to exclude test points with inssuficient counts of an specific MHC type')
    return parser.parse_args()


# Parse the command-line arguments and store them in the 'args' variable
args = args_parser()
args_dict = vars(args)

# Reading test results
df_results = pd.read_csv(args_dict['test_results'])

# Filtering test results according to binding affinity
if args_dict['ba_filter'] == True:
    print(f"\nFiltering test results according to binding affinity >= {args_dict['ba_thr']} to exclude not binder peptides.")
    print(f"Number of rows before filtering: {len(df_results)}")
    df_results = df_results[df_results['BA'] >= args_dict['ba_thr']]
    print(f"Number of rows after filtering: {len(df_results)}")

# Extract original basename of the input file by the ID introduced
file_name = os.path.basename(args_dict['test_results'])
start_index = file_name.find(args_dict['file_id'])
# Extract the part starting from the file_id
if args_dict['keep_bn'] == True:
    if start_index != -1:
        basename = file_name[start_index:-4]
    else:
        basename = ''
        print('\n---------------------------------------------------------------------------------')
        print(f"NOTE: Substring {args_dict['file_id']} not found in the original basename.")
        print('No ID will be included to the new processed outputs.')
        print('---------------------------------------------------------------------------------\n')
else:
    basename = ''
    print('\n---------------------------------------------------------------------------------')
    print("NOTE: No ID will be included to the new processed outputs.")
    print('---------------------------------------------------------------------------------\n')

# Function to get the rank of the highest predicted values for each MHC
def top_results(df, n_mhc, outdir='', rank_val = 200):
    
    # Create the output directory according to the basename (if it is a valid one)
    if outdir != '':
        outdir = outdir + '_MHCresults'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    else:
        # Create the output directory with the current date as the name
        current_date = datetime.now().strftime("%y%m%d-%H·%M·%S")
        outdir = current_date + '_MHCresults'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    # Eliminate the rows that have an MHC type with low counts (n_mhc)
    mhc_counts = df.groupby('MHC_name')['MHC_name'].transform('count')
    df = df[mhc_counts >= n_mhc]
    print(f"Number of rows after eliminating rows with MHC types < {n_mhc}: {len(df)}")

    # Get unique MHC names from the DataFrame and loop according to it
    unique_mhc_names = df['MHC_name'].unique()
    for mhc in unique_mhc_names:
        
        # Filter the DataFrame to include only the specified MHC
        mhc_df = df[df['MHC_name'] == mhc]

        # Get the highest predicted values according to the predictions
        sorted_df = mhc_df.sort_values(by='pred', ascending=False)
        top_rank = sorted_df.head(rank_val)
        
        # Return the complete csv
        out_name = os.path.join(outdir, 'top_' + str(rank_val) + '_' + mhc + '_' + basename + '.csv')
        top_rank.to_csv(out_name, index=False)

        # Return only the motif list
        motif_df = top_rank[['motif']]
        outmotif_name = os.path.join(outdir, 'top_' + str(rank_val) + '_motif_' + mhc + '_' + basename + '.txt')
        motif_df.to_csv(outmotif_name, sep='\t', index=False, header=False)

# Function to get the rank of the highest predicted values for a concrete MHC
def top_results_mhc(df, mhc='DRB1_0101', rank_val = 200):
    
    # Get the highest predicted values according to the predictions for an specific MHC molecule
    filtered_df = df[df['MHC_name'] == mhc]
    sorted_df = filtered_df.sort_values(by='pred', ascending=False)
    top_rank = sorted_df.head(rank_val)
    
    # Return the complete csv
    out_name = 'top_' + str(rank_val) + '_' + mhc + '_' + basename + '.csv'
    top_rank.to_csv(out_name, index=False)

    # Return only the motif list
    motif_df = top_rank[['motif']]
    outmotif_name = 'top_' + str(rank_val) + '_motif_' + '_' + mhc + '_' + basename + '.txt'
    motif_df.to_csv(outmotif_name, sep='\t', index=False, header=False)


# Obtention of the highest predicted values (taking into account all possible MHC)
top_results(df_results, n_mhc = args_dict['mhc_count'], outdir = basename, rank_val = args_dict['rank'])

# Obtention of the highest predicted values for 1 specific MHC
if args_dict['add_1mhc'] == True:
    top_results_mhc(df_results, mhc = args_dict['mhc_unique'], rank_val = args_dict['rank'])