import copy

import pandas as pd
import numpy as np
import torch
import multiprocessing
import math
from torch.utils.data import TensorDataset
from src.utils import pkl_load, pkl_dump
import os
import warnings
import peptides

warnings.filterwarnings('ignore')

DATADIR = '/Users/riwa/Documents/code/PyNNalign/data/' if os.path.exists(os.path.abspath('/Users/riwa/Documents/code/PyNNalign/data')) else '/home/projects/vaccine/people/yatwan/PyNNalign/data/'
OUTDIR = '/Users/riwa/Documents/code/PyNNalign/output/' if os.path.exists(os.path.abspath('/Users/riwa/Documents/code/PyNNalign/output')) else '/home/projects/vaccine/people/yatwan/PyNNalign/output/'
# Stupid hardcoded variable
CNN_FEATS = ['EL_ratio', 'anchor_mutation', 'delta_VHSE1', 'delta_VHSE3', 'delta_VHSE7', 'delta_VHSE8',
             'delta_aliphatic_index',
             'delta_boman', 'delta_hydrophobicity', 'delta_isoelectric_point', 'delta_rank']


def _init(DATADIR):
    #### ==== CONST (blosum, multiprocessing, keys, etc) ==== ####
    VAL = math.floor(4 + (multiprocessing.cpu_count() / 1.5))
    N_CORES = VAL if VAL <= multiprocessing.cpu_count() else int(multiprocessing.cpu_count() - 2)

    MATRIXDIR = f'{DATADIR}Matrices/'
    ICSDIR = f'{DATADIR}ic_dicts/'
    AA_KEYS = [x for x in 'ARNDCQEGHILKMFPSTWYV']

    CHAR_TO_INT = dict((c, i) for i, c in enumerate(AA_KEYS))
    INT_TO_CHAR = dict((i, c) for i, c in enumerate(AA_KEYS))
    CHAR_TO_INT['X'] = -1
    CHAR_TO_INT['-'] = -1
    INT_TO_CHAR[-1] = '-'

    BG = np.loadtxt(f'{MATRIXDIR}bg.freq.fmt', dtype=float)
    BG = dict((k, v) for k, v in zip(AA_KEYS, BG))

    # BLOSUMS 50
    BL50 = {}
    _blosum50 = np.loadtxt(f'{MATRIXDIR}BLOSUM50', dtype=float).T
    for i, letter_1 in enumerate(AA_KEYS):
        BL50[letter_1] = {}
        for j, letter_2 in enumerate(AA_KEYS):
            BL50[letter_1][letter_2] = _blosum50[i, j]
    BL50_VALUES = {k: np.array([v for v in BL50[k].values()]) for k in BL50}
    # BLOSUMS 62
    BL62_DF = pd.read_csv(f'{MATRIXDIR}BLOSUM62', sep='\s+', comment='#', index_col=0)
    BL62 = BL62_DF.to_dict()
    BL62_VALUES = BL62_DF.drop(columns=['B', 'Z', 'X', '*'], index=['B', 'Z', 'X', '*'])
    BL62_VALUES = dict((x, BL62_VALUES.loc[x].values) for x in BL62_VALUES.index)

    # BLOSUMS 62 FREQS
    _blosum62 = np.loadtxt(f'{MATRIXDIR}BLOSUM62.freq_rownorm', dtype=float).T
    BL62FREQ = {}
    BL62FREQ_VALUES = {}
    for i, letter_1 in enumerate(AA_KEYS):
        BL62FREQ[letter_1] = {}
        BL62FREQ_VALUES[letter_1] = _blosum62[i]
        for j, letter_2 in enumerate(AA_KEYS):
            BL62FREQ[letter_1][letter_2] = _blosum62[i, j]
    ICS_KL = pkl_load(ICSDIR + 'ics_kl_new.pkl')
    ICS_SHANNON = pkl_load(ICSDIR + 'ics_shannon.pkl')
    HLAS = ICS_SHANNON[9].keys()

    return VAL, N_CORES, DATADIR, AA_KEYS, CHAR_TO_INT, INT_TO_CHAR, BG, BL62FREQ, BL62FREQ_VALUES, BL50, BL50_VALUES, BL62, BL62_VALUES, HLAS, ICS_KL, ICS_SHANNON


VAL, N_CORES, DATADIR, AA_KEYS, CHAR_TO_INT, INT_TO_CHAR, BG, BL62FREQ, BL62FREQ_VALUES, BL50, BL50_VALUES, BL62, BL62_VALUES, HLAS, ICS_KL, ICS_SHANNON = _init(
    DATADIR)

encoding_matrix_dict = {'onehot': None,
                        'BL62LO': BL62_VALUES,
                        'BL62FREQ': BL62FREQ_VALUES,
                        'BL50LO': BL50_VALUES}
ics_dict = {'KL': ICS_KL,
            'Shannon': ICS_SHANNON}


######################################
####      assertion / checks      ####
######################################

def verify_df(df, seq_col, hla_col, target_col):
    df = copy.deepcopy(df)
    unique_labels = sorted(df[target_col].dropna().unique())
    # Checks binary label
    assert ([int(x) for x in sorted(unique_labels)]) in [[0, 1], [0], [1]], f'Labels are not 0, 1! {unique_labels}'
    # Checks if any seq not in alphabet
    try:
        df = df.drop(df.loc[df[seq_col].apply(lambda x: any([z not in AA_KEYS and not z == '-' for z in x]))].index)
    except:
        print(len(df), df.columns, seq_col, AA_KEYS)
        raise ValueError
    # Checks if HLAs have correct format
    if all(df[hla_col].apply(lambda x: not x.startswith('HLA-'))):
        df[hla_col] = df[hla_col].apply(lambda x: 'HLA-' + x)
    df[hla_col] = df[hla_col].apply(lambda x: x.replace('*', '').replace(':', ''))
    # Check HLA only in subset
    try:
        df = df.query(f'{hla_col} in @HLAS')
    except:
        print(type(df), type(HLAS), HLAS, hla_col)
        raise ValueError(f'{type(df)}, {type(HLAS)}, {HLAS}, {hla_col}, {df[hla_col].unique()}')

    return df


def assert_encoding_kwargs(encoding_kwargs, mode_eval=False):
    """
    Assertion / checks for encoding kwargs and verify all the necessary key-values
    are in
    """
    # Making a deep copy since dicts are mutable between fct calls
    encoding_kwargs = copy.deepcopy(encoding_kwargs)
    if encoding_kwargs is None:
        encoding_kwargs = {'max_len': 12,
                           'encoding': 'onehot',
                           'standardize': False}
    essential_keys = ['max_len', 'encoding', 'standardize']
    keys_check = [x in encoding_kwargs.keys() for x in essential_keys]
    keys_check_dict = {k: v for (k, v) in zip(essential_keys, keys_check) if v == False}
    assert all(keys_check), f'Encoding kwargs don\'t contain the essential key-value pairs! ' \
                            f"{list(keys_check_dict.keys())} are missing!"

    if mode_eval:
        if any([(x not in encoding_kwargs.keys()) for x in ['seq_col', 'hla_col', 'target_col', 'rank_col']]):
            if 'seq_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'seq_col': 'icore_mut'})
            if 'hla_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'hla_col': 'HLA'})
            if 'target_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'target_col': 'agg_label'})
            if 'rank_col' not in encoding_kwargs.keys():
                encoding_kwargs.update({'rank_col': 'EL_rank_mut'})

        # This KWARGS not needed in eval mode since I'm using Pipeline and Pipeline
        del encoding_kwargs['standardize']
    return encoding_kwargs


######################################
####      SEQUENCES ENCODING      ####
######################################


def get_aa_properties(df, seq_col='icore_mut', do_vhse=True, prefix=''):
    """
    Compute some AA properties that I have selected
    keep = ['aliphatic_index', 'boman', 'hydrophobicity',
        'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']
    THIS KEEP IS BASED ON SOME FEATURE DISTRIBUTION AND CORRELATION ANALYSIS
    Args:
        df (pandas.DataFrame) : input dataframe, should contain at least the peptide sequences
        seq_col (str) : column name containing the peptide sequences

    Returns:
        out (pandas.DataFrame) : The same dataframe but + the computed AA properties

    """
    out = df.copy()

    out[f'{prefix}aliphatic_index'] = out[seq_col].apply(lambda x: peptides.Peptide(x).aliphatic_index())
    out[f'{prefix}boman'] = out[seq_col].apply(lambda x: peptides.Peptide(x).boman())
    out[f'{prefix}hydrophobicity'] = out[seq_col].apply(lambda x: peptides.Peptide(x).hydrophobicity())
    out[f'{prefix}isoelectric_point'] = out[seq_col].apply(lambda x: peptides.Peptide(x).isoelectric_point())
    # out['PD2'] = out[seq_col].apply(lambda x: peptides.Peptide(x).physical_descriptors()[1])
    # out['charge_7_4'] = out[seq_col].apply(lambda x: peptides.Peptide(x).charge(pH=7.4))
    # out['charge_6_65'] = out[seq_col].apply(lambda x: peptides.Peptide(x).charge(pH=6.65))
    if do_vhse:
        vhse = out[seq_col].apply(lambda x: peptides.Peptide(x).vhse_scales())
        # for i in range(1, 9):
        #     out[f'VHSE{i}'] = [x[i - 1] for x in vhse]
        for i in [1, 3, 7, 8]:
            out[f'VHSE{i}'] = [x[i - 1] for x in vhse]

    # Some hardcoded bs
    return out, ['aliphatic_index', 'boman', 'hydrophobicity',
                 'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']


def encode(sequence, max_len=None, encoding='onehot', pad_scale=None):
    """
    encodes a single peptide into a matrix, using 'onehot' or 'blosum'
    if 'blosum', then need to provide the blosum dictionary as argument
    """
    assert encoding in encoding_matrix_dict.keys(), f'Wrong encoding key {encoding} passed!'\
                                                    f'Should be any of {encoding_matrix_dict.keys()}'
    # One hot encode by setting 1 to positions where amino acid is present, 0 elsewhere
    size = len(sequence)
    blosum_matrix = encoding_matrix_dict[encoding]
    if encoding == 'onehot':
        int_encoded = [CHAR_TO_INT[char] for char in sequence]
        onehot_encoded = list()
        for value in int_encoded:
            letter = [0 for _ in range(len(AA_KEYS))]
            letter[value] = 1 if value != -1 else 0
            onehot_encoded.append(letter)
        tmp = np.array(onehot_encoded)

    # BLOSUM encode
    else:
        if blosum_matrix is None or not isinstance(blosum_matrix, dict):
            raise Exception('No BLOSUM matrix provided!')

        tmp = np.zeros([size, len(AA_KEYS)], dtype=np.float32)
        for idx in range(size):
            # Here, the way Morten takes cares of Xs is to leave it blank, i.e. as zeros
            # So only use blosum matrix to encode if sequence[idx] != 'X'
            if sequence[idx]!='X':
                tmp[idx, :] = blosum_matrix[sequence[idx]]


    # Padding if max_len is provided
    if max_len is not None and max_len > size:
        diff = int(max_len) - int(size)
        try:
            if pad_scale is None:
                pad_scale = 0 if encoding=='onehot' else -12
            tmp = np.concatenate([tmp, pad_scale*np.ones([diff, len(AA_KEYS)], dtype=np.float32)],
                                 axis=0)
        except:
            print('Here in encode', type(tmp), tmp.shape, len(AA_KEYS), type(diff), type(max_len), type(size), sequence)
            #     return tmp, diff, len(AA_KEYS)
            raise Exception
    return torch.from_numpy(tmp).float()


def encode_batch(sequences, max_len=None, encoding='onehot', pad_scale=None):
    """
    Encode multiple sequences at once.
    """
    if max_len is None:
        max_len = max([len(x) for x in sequences])
    # Contiguous to allow for .view operation
    return torch.stack([encode(seq, max_len, encoding, pad_scale) for seq in sequences]).contiguous()


def onehot_decode(onehot_sequence):
    if type(onehot_sequence) == np.ndarray:
        return ''.join([INT_TO_CHAR[x.item()] for x in onehot_sequence.nonzero()[1]])
    elif type(onehot_sequence) == torch.Tensor:
        return ''.join([INT_TO_CHAR[x.item()] for x in onehot_sequence.nonzero()[:, 1]])


def onehot_batch_decode(onehot_sequences):
    return np.stack([onehot_decode(x) for x in onehot_sequences])


def get_ic_weights(df, ics_dict: dict, max_len=None, seq_col='Peptide', hla_col='HLA', mask=False,
                   invert=False, threshold=0.234):
    """

    Args:
        df:
        ics_dict:
        max_len:
        seq_col:
        hla_col:
        invert: Invert the behaviour; for KL/Shannon, will take IC instead of 1-IC as weight
                For Mask, will amplify MIA positions (by 1.3) instead of setting to 0

    Returns:

    """
    # if 'len' not in df.columns:

    df['len'] = df[seq_col].apply(len)
    if max_len is not None:
        df = df.query('len<=@max_len')
    else:
        max_len = df['len'].max()
    # Weighting the encoding wrt len and HLA
    lens = df['len'].values
    pads = [max_len - x for x in lens]
    hlas = df[hla_col].str.replace('*', '').str.replace(':', '').values
    # If mask is true, then the weight is just a 0-1 mask filter
    # Using the conserved / MIAs positions instead of the ICs
    if mask:
        # Get mask for where the values should be thresholded to 0 and 1
        weights = np.stack([np.pad(ics_dict[l][hla][0.25], pad_width=(0, pad), constant_values=(1, 1)) \
                            for l, hla, pad in zip(lens, hlas, pads)])
        # IC > 0.3 goes to 0 because anchor position
        # IC <= 0.3 goes to 1 because "MIA" position
        idx_min = (weights > threshold)
        idx_max = (weights <= threshold)
        if invert:
            weights[idx_min] = 1
            weights[idx_max] = 0
        else:
            weights[idx_min] = 0
            weights[idx_max] = 1

    else:
        if invert:  # If invert, then uses the actual IC as weight
            weights = np.stack([np.pad(ics_dict[l][hla][0.25], pad_width=(0, pad), constant_values=(0, 1)) \
                                for l, hla, pad in zip(lens, hlas, pads)])
        else:  # Else we get the weight with the 1-IC depending on the IC dict provided
            weights = 1 - np.stack([np.pad(ics_dict[l][hla][0.25], pad_width=(0, pad), constant_values=(1, 1)) \
                                    for l, hla, pad in zip(lens, hlas, pads)])

    weights = np.expand_dims(weights, axis=2).repeat(len(AA_KEYS), axis=2)
    return weights


# Here stuff for extra AA bulging out:

def find_extra_aa(core, icore):
    """
    Finds the bulging out AA between an icore and its corresponding core, returning the extra AA as "frequencies"
    Args:
        core:
        icore:

    Returns:

    """
    assert len(core) == 9, f'Core is not of length 9 somehow: {core}'
    if len(icore) == len(core) or len(icore) == 8:
        return np.zeros((20)), np.array(0)

    elif len(icore) > len(core):
        results = []
        j = 0
        for i, char in enumerate(icore):
            if char != core[j]:
                results.append(char)
            else:
                j += 1
        # Here changed to len icore - len core to get len of bulge
        # return (encode(''.join(results)).sum(axis=0).numpy() / (len(icore)-len(core))).astype(np.float32)

        # Here, changed to return the extra + the length so that we can do the weighted division
        return encode(''.join(results)).sum(axis=0).numpy(), np.array(len(icore) - len(core))


def batch_find_extra_aa(core_seqs, icore_seqs):
    """
    Same as above but by batch
    Args:
        core_seqs:
        icore_seqs:

    Returns:

    """
    mapped = list(map(find_extra_aa, core_seqs, icore_seqs))
    encoded, lens = np.array([x[0] for x in mapped]), np.array([x[1] for x in mapped])
    return encoded, lens


def encode_batch_weighted(df, ics_dict=None, device=None, max_len=None, encoding='onehot', seq_col='Peptide',
                          hla_col='HLA', target_col='agg_label', mask=False, invert=False, threshold=.234,
                          return_weights=False):
    """
    Takes as input a df containing sequence, len, HLA;
    Batch onehot-encode all sequences & weights them with (1-IC) depending on the ICs dict given

    Args:
        target_col:
        df (pandas.DataFrame): DF containing pep sequence, HLA, optionally 'len'
        ics_dict (dict): Dictionary containing the ICs
        device (str) : device for cpu or cuda transfer
        max_len (int): Maximum length to consider
        encoding (str) : 'onehot' or 'blosum'
        seq_col (str): Name of the column containing the Peptide sequences (default = 'Peptide')
        hla_col (str): Name of the column containing the HLA alleles (default = 'HLA')


    Returns:
        weighted_sequence (torch.Tensor): Tensor containing the weighted onehot-encoded peptide sequences.
    """
    df = verify_df(df, seq_col, hla_col, target_col)
    if seq_col == 'expanded_input':
        df['seq_len'] = df[seq_col].apply(lambda x: len(x) - x.count('-'))
    else:
        df['seq_len'] = df[seq_col].apply(len)
    if max_len is not None:
        df = df.query('seq_len<=@max_len')
    else:
        max_len = df['seq_len'].max()

    # Encoding the sequences
    encoded_sequences = encode_batch(df[seq_col].values, max_len, encoding=encoding)
    if ics_dict is not None:
        weights = get_ic_weights(df, ics_dict, max_len, seq_col, hla_col, mask, invert, threshold)
    else:
        # Here, if no ics_dict is provided, the normal weight will just be ones everywhere
        # In case we are not doing weighted sequence (either for onehot-input or frequency computation)
        weights = np.ones(encoded_sequences.shape)
    weighted_sequences = torch.from_numpy(weights) * encoded_sequences
    true_lens = df['seq_len'].values
    if device is None:
        return weighted_sequences.float(), true_lens
    else:
        return weighted_sequences.to(device).float(), true_lens


def get_train_valid_dfs(dataframe, fold_inner, fold_outer):
    train_data = dataframe.query('fold != @fold_inner and fold != @fold_outer')
    valid_data = dataframe.query('fold == @fold_inner')
    return train_data, valid_data


def get_array_dataset(df, ics_dict, max_len=12, encoding='onehot', seq_col='icore_mut', hla_col='HLA',
                      target_col='agg_label', rank_col='EL_rank_mut', mask=False, invert=False, add_rank=True,
                      add_aaprop=False, remove_pep=False, threshold=0.234, icore_bulge=False, core_col='core_mut',
                      icore_col='icore_mut'):
    """
        Computes the frequencies as the main features
        Takes as input a df containing sequence, len, HLA;
        Batch encode all sequences & weights them with (1-IC) depending on the ICs dict given
        Stacks it with the targets in another dimension, return as a np.ndarray
        Big mess of a fct to be honest :-)
    Args:
        df:
        ics_dict:
        max_len:
        encoding:
        blosum_matrix:
        seq_col:
        hla_col:
        target_col:

        add_rank:
        mask:
        add_aaprop:
        remove_pep: Boolean switch to discard the AA sequence/freq in features (e.g. keep only rank or only chem props)

    Returns:
        tensor_dataset (torch.utils.data.TensorDataset): Dataset containing the tensors X and y
    """
    # df = verify_df(df, seq_col, hla_col, target_col)
    encoded_weighted, true_lens = encode_batch_weighted(df, ics_dict, 'cpu', max_len, encoding, seq_col, hla_col,
                                                        target_col, mask, invert, threshold)
    x = batch_compute_frequency(encoded_weighted.numpy(), true_lens)
    if add_rank:
        ranks = np.expand_dims(df[rank_col].values, 1)
        x = np.concatenate([x, ranks], axis=1)

    if add_aaprop:
        # New way of doing it already  saves the aa props to the df to
        # not re-compute them everytime, here for now because I
        if all([x in df.columns for x in ['aliphatic_index', 'boman', 'hydrophobicity',
                                          'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']]):
            aa_props = df[['aliphatic_index', 'boman', 'hydrophobicity',
                           'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']].values
        else:
            df_props, columns = get_aa_properties(df, seq_col)
            aa_props = df[columns].values

        x = np.concatenate([x, aa_props], axis=1)

    y = df[target_col].values
    # Queries whatever is above 20, and only keeps that as feature
    if remove_pep and (add_rank or add_aaprop):
        x = x[:, 20:]
    return x, y


def get_dataset(df, ics_dict, max_len=12, encoding='onehot', seq_col='icore_mut', hla_col='HLA', target_col='agg_label',
                rank_col='EL_rank_mut', mut_col=None, adaptive=False, mask=False, invert=False, add_rank=False,
                add_aaprop=False, remove_pep=False, mask_aa=None, threshold=.234, icore_bulge=False,
                core_col='core_mut', icore_col='icore_mut'):
    """
    """
    # df = verify_df(df, seq_col, hla_col, target_col)

    # Def arguments because we only use add rank
    if adaptive:
        # Splitting into anchor_mutation groups
        anchors = df.query('anchor_mutation==True')
        non_ancs = df.query('anchor_mutation==False')
        # Here Invert is true (so the anchors get IC instead of 1-IC)
        x_anchors, y_anchors = get_array_dataset(anchors, ics_dict, max_len, encoding, seq_col, hla_col, target_col,
                                                 invert=True, add_rank=True, add_aaprop=False, remove_pep=False)
        # Adding the mut columns and concatenating on columns axis (ax=1)
        if len(mut_col) > 0:
            mut_anchors = anchors[mut_col].values
            x_anchors = np.concatenate([x_anchors, mut_anchors], axis=1)
        # Here, invert is False (so using 1-IC, to up-weigh non-anchor positions for non anc mutations
        x_non, y_non = get_array_dataset(non_ancs, ics_dict, max_len, encoding, seq_col, hla_col, target_col,
                                         invert=False, add_rank=True, add_aaprop=False, remove_pep=False)
        # Same
        if mut_col is not None and type(mut_col) == list:
            if len(mut_col) > 0:
                mut_non = non_ancs[mut_col].values
                x_non = np.concatenate([x_non, mut_non], axis=1)
        # Joining the two Xs and Ys into single x,y vectors
        x = np.concatenate([x_anchors, x_non], axis=0)
        y = np.concatenate([y_anchors, y_non], axis=0)

    else:
        x, y = get_array_dataset(df, ics_dict, max_len, encoding, seq_col, hla_col, target_col, rank_col, mask, invert,
                                 add_rank=add_rank, add_aaprop=add_aaprop, remove_pep=remove_pep, threshold=threshold,
                                 icore_bulge=icore_bulge, core_col=core_col, icore_col=icore_col)
        if mut_col is not None and type(mut_col) == list:
            if len(mut_col) > 0:
                mut_scores = df[mut_col].values
                x = np.concatenate([x, mut_scores], axis=1)
    if mask_aa:
        if mask_aa.lower() == 'false':
            return x, y
        assert mask_aa.upper() in AA_KEYS, f'Amino acid to mask is not in the AA alphabet!' \
                                           '\nYou provided {mask_aa}, and the alphabet is {AA_KEYS}'
        x[:, AA_KEYS.index(mask_aa.upper())] = 0
    return x, y