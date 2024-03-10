import copy

import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
import multiprocessing
import math
from torch.utils.data import TensorDataset
from src.utils import pkl_load, pkl_dump
import os
import warnings

warnings.filterwarnings('ignore')

DATADIR = '/Users/riwa/Documents/code/PyNNalign/data/' if os.path.exists(
    os.path.abspath('/Users/riwa/Documents/code/PyNNalign/data')) else '../data/'
OUTDIR = '/Users/riwa/Documents/code/PyNNalign/output/' if os.path.exists(
    os.path.abspath('/Users/riwa/Documents/code/PyNNalign/output')) else '../output/'
# Stupid hardcoded variable
CNN_FEATS = ['EL_ratio', 'anchor_mutation', 'delta_VHSE1', 'delta_VHSE3', 'delta_VHSE7', 'delta_VHSE8',
             'delta_aliphatic_index',
             'delta_boman', 'delta_hydrophobicity', 'delta_isoelectric_point', 'delta_rank']


def _init(DATADIR):
    VAL = math.floor(4 + (multiprocessing.cpu_count() / 1.5))
    N_CORES = VAL if VAL <= multiprocessing.cpu_count() else int(multiprocessing.cpu_count() - 2)

    MATRIXDIR = f'{DATADIR}Matrices/'
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
    # TODO read pseudoseq here
    with open(f'{MATRIXDIR}MHC_pseudo.dat', 'r') as f:
        # lines = [x.rstrip('\n').split('\t') for x in f.readlines()]
        lines = [x.replace(' ', ';').rstrip('\n').replace('\t', ';') for x in f.readlines()]
        # extra chars
        replaced = [[x for x in z.split(';') if len(x) > 0] for z in lines]
        PSEUDOSEQDICT = {k: v for k, v in replaced}
    return VAL, N_CORES, DATADIR, AA_KEYS, CHAR_TO_INT, INT_TO_CHAR, BG, BL62FREQ, BL62FREQ_VALUES, BL50, BL50_VALUES, BL62, BL62_VALUES, PSEUDOSEQDICT


VAL, N_CORES, DATADIR, AA_KEYS, CHAR_TO_INT, INT_TO_CHAR, BG, BL62FREQ, BL62FREQ_VALUES, BL50, BL50_VALUES, BL62, BL62_VALUES, PSEUDOSEQDICT = _init(
    DATADIR)

encoding_matrix_dict = {'onehot': None,
                        'BL62LO': BL62_VALUES,
                        'BL62FREQ': BL62FREQ_VALUES,
                        'BL50LO': BL50_VALUES}


######################################
####      SEQUENCES ENCODING      ####
######################################


def encode(sequence, max_len=None, encoding='onehot', pad_scale=None):
    """
    encodes a single peptide into a matrix, using 'onehot' or 'blosum'
    if 'blosum', then need to provide the blosum dictionary as argument
    """
    assert encoding in encoding_matrix_dict.keys(), f'Wrong encoding key {encoding} passed!' \
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
            if sequence[idx] != 'X' and sequence[idx] != '-':
                tmp[idx, :] = blosum_matrix[sequence[idx]]

    # Padding if max_len is provided
    if max_len is not None and max_len > size:
        diff = int(max_len) - int(size)
        try:
            if pad_scale is None:
                pad_scale = 0 if encoding == 'onehot' else -12
            tmp = np.concatenate([tmp, pad_scale * np.ones([diff, len(AA_KEYS)], dtype=np.float32)],
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


def _get_pfr_mask_after(l, max_len, window_size):
    """
        Gets the PFR lengths mask of the positions of PFR regions after the kmer
        Split from the functions below because this can be used in get_fr_lengths
        and also used to get the indices in
    """

    n_windows = max_len - window_size + 1
    # For a given length, this determins the amount of windows with 3 AA after
    n_3 = l - window_size - 2
    # Then the number of windows with 2 and 1 AAs are computed from the previous
    n_2 = 1 if n_3 >= 0 else max(n_3 + 1, -1)
    n_1 = 1 if n_2 >= 0 else max(n_2 + 1, -1)
    # Here we create the mask to of the N aas / windows
    # ex if max_len=13, l=12, window_size=9
    # we will get [3, 2, 1, 0, 0]
    # for l = 11, we get [2, 1, 0, 0, 0] etc.
    mask = torch.cat([torch.full((max(n_3, 0),), 3),
                      torch.full((max(n_2, 0),), 2),
                      torch.full((max(n_1, 0),), 1)])
    return F.pad(mask, (0, n_windows - len(mask)), value=0)


def _get_pfr_mask_after_batch(lengths, max_len, window_size):
    return torch.stack([_get_pfr_mask_after(l, max_len, window_size) for l in lengths], dim=0)


def _get_pfr_indices_after(l, max_len, window_size):
    """
    Gets the indices (to slice) of the positions of PFR regions after the kmer
    """
    n_windows = max_len - window_size + 1
    mask = _get_pfr_mask_after(l, max_len, window_size)
    # Then we convert the mask into a pair of indices indicating the positional index of the aas to select
    return torch.stack([torch.arange(0, n_windows) + window_size,
                        torch.arange(0, n_windows) + window_size + mask]).T


def _get_pfr_indices_after_batch(lengths, max_len, window_size):
    # batching
    return torch.stack([_get_pfr_indices_after(l, max_len, window_size) for l in lengths])


def _mask_from_index(index_tensor, max_len):
    # from an index tensor, create the mask to mask the input tensor ;
    N, S, _ = index_tensor.shape  # Batch size, number of sequences, 2 (for start and end indices)

    # Create a range tensor of shape [1, 1, max_len] for broadcasting
    range_tensor = torch.arange(max_len).unsqueeze(0).unsqueeze(0)

    # Expand before_indices for start and end to match the shape for broadcasting
    start_indices = index_tensor[:, :, 0].unsqueeze(2)  # Shape: [N, S, 1]
    end_indices = index_tensor[:, :, 1].unsqueeze(2)  # Shape: [N, S, 1], +1 to include the end index in the range

    # Create the binary mask
    mask = (range_tensor >= start_indices) & (range_tensor < end_indices)  # Shape: [N, S, max_len]

    return mask.int()


def get_pfr_values(sequences, max_len, window_size, indel=False):
    # Get the true lengths of each sequence to create the masks
    len_tensor = torch.tensor([len(x) for x in sequences])
    n_windows = max_len - window_size + 1
    # Get the data vector to repeat and mask to compute PFR
    blosum_freq = encode_batch(sequences, max_len, 'BL62FREQ', None)
    # Get the repeated blosum_freq vector (creating N_windows copies along dim=1)
    repeats = blosum_freq.unsqueeze(1).repeat(1, n_windows, 1, 1)
    # The before mask is the same no matter the length so create it and just repeat it to get the indices
    before_mask = torch.full((n_windows,), 3)
    before_mask[:3] = torch.tensor([0, 1, 2], dtype=torch.float32)
    before_indices = torch.stack([torch.arange(0, n_windows) - before_mask, torch.arange(0, n_windows)]).T.repeat(
        len(sequences), 1, 1)
    # Use my custom function and length tensor to create the after_indices
    after_indices = _get_pfr_indices_after_batch(len_tensor, max_len, window_size)
    # Create the mask of shape (N, n_windows, max_len, 20) to use on the repeated freq_data and broadcast it 20 times along amino acid dimension
    before_mask = _mask_from_index(before_indices, max_len).unsqueeze(-1).repeat(1, 1, 1, 20)
    after_mask = _mask_from_index(after_indices, max_len).unsqueeze(-1).repeat(1, 1, 1, 20)
    # Take dim=2 because that's the sequence length dimension
    pfrs = torch.cat([(repeats * before_mask).mean(dim=2),
                      (repeats * after_mask).mean(dim=2)], dim=2)
    if indel:
        pfrs = torch.cat([pfrs, torch.zeros(len(pfrs), window_size+1, pfrs.shape[-1])], dim=1)
    return pfrs


def _get_after_range(length, max_len, window_size):
    n_windows = max_len - window_size + 1
    range_tensor = torch.arange(max(0, length - window_size), 0, -1, dtype=torch.float32)
    # Pad with 0s up to n_windows
    return F.pad(range_tensor, (0, n_windows - len(range_tensor))).unsqueeze(0).T


def get_fr_lengths(len_tensor, max_len, window_size, indel=False):
    n_windows = max_len - window_size + 1
    # FR lengths before are always the same
    fr_lengths_before = (torch.arange(n_windows) / (torch.arange(n_windows) + 1)).unsqueeze(0).T.unsqueeze(0)
    fr_lengths_before = fr_lengths_before.repeat(len(len_tensor), 1, 2)
    fr_lengths_before[:, :, 1] = 1 - fr_lengths_before[:, :, 1]
    # FR lengths after depend on the actual length after
    fr_lengths_after = torch.stack([_get_after_range(l, max_len, window_size) for l in len_tensor])
    fr_lengths_after /= (fr_lengths_after + 1)
    fr_lengths_after = fr_lengths_after.repeat(1, 1, 2)
    fr_lengths_after[:, :, 1] = 1 - fr_lengths_after[:, :, 1]
    fr_lengths = torch.cat([fr_lengths_before, fr_lengths_after], dim=2)
    if indel:
        fr_lengths = torch.cat([fr_lengths, torch.zeros(len(fr_lengths), window_size+1, fr_lengths.shape[-1])], dim=1)
    return fr_lengths


def get_pep_len_onehot(len_tensor, max_len, window_size, min_clip, max_clip, indel=False):
    # Scaling the lengths and clipping to set to min/max clip values
    scaled_lengths = (len_tensor - min_clip + 1).clip(min=0, max=max_clip - min_clip + 1)
    # getting onehot encoding and repeating to accomodate n_windows
    # In case of indels, we get n_windows = (max_len - window_size + 1) + (window_size+1) = max_len + 2
    n_windows = max_len + 2 if indel else max_len - window_size + 1
    return F.one_hot(scaled_lengths, num_classes=max_clip - min_clip + 2).unsqueeze(1).repeat(1, n_windows, 1)


def get_indel_windows(sequence, window_size):
    """
    From one sequence (string), expand into a list of available windows with either insertions or deletions
    """
    length = len(sequence)
    indel_windows = []

    # Insertion for sequences shorter than the window size
    if length < window_size:
        for i in range(window_size):
            indel_windows.append(sequence[:i] + '-' + sequence[i:])
        indel_windows.append('-' * window_size)
        # Replicate sequence for sequences equal to the window size
    elif length == window_size:
        indel_windows.append(sequence)
        while len(indel_windows) < (window_size + 1):
            indel_windows.append('-' * window_size)
            # Deletion for sequences longer than the window size
    else:
        del_len = length - window_size
        for i in range(length - del_len + 1):
            indel_windows.append(sequence[:i] + sequence[i + del_len:])

    return indel_windows


def do_insertion_deletion(sequence, max_len=13, encoding='BL50LO', pad_scale=-20, window_size=9):
    """
    Take a sequence and expand the windows then batch_encode them
    """
    indel_windows = get_indel_windows(sequence, window_size)
    # Encoding the sequences
    encoded_sequences = encode_batch(indel_windows, max_len=max_len, encoding=encoding, pad_scale=pad_scale)
    return encoded_sequences


def batch_insertion_deletion(sequences, max_len=13, encoding='BL50LO', pad_scale=-20, window_size=9):
    return torch.stack(
        [do_insertion_deletion(seq, max_len=max_len, encoding=encoding, pad_scale=pad_scale, window_size=window_size)
         for
         seq in sequences])


def create_indel_mask(length, window_size):
    mask = torch.zeros(1, window_size + 1, 1)
    if length < window_size:
        mask[:, :-1, :].fill_(1)
    elif length == window_size:
        # Actually shouldn't fill with 1 because we are concatenating to the other mask
        # and only a single window (i.e. the first, un-concatenated one) is the correct one
        # mask[:, :1, :].fill_(1)
        pass
    elif length > window_size:
        mask.fill_(1)
    return mask


def batch_indel_mask(lengths, window_size):
    return torch.cat([create_indel_mask(length, window_size) for length in lengths], dim=0)


####################################################
#     OLD Here stuff for extra AA bulging out:     #
####################################################

#
# def get_aa_properties(df, seq_col='icore_mut', do_vhse=True, prefix=''):
#     """
#     Compute some AA properties that I have selected
#     keep = ['aliphatic_index', 'boman', 'hydrophobicity',
#         'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']
#     THIS KEEP IS BASED ON SOME FEATURE DISTRIBUTION AND CORRELATION ANALYSIS
#     Args:
#         df (pandas.DataFrame) : input dataframe, should contain at least the peptide sequences
#         seq_col (str) : column name containing the peptide sequences
#
#     Returns:
#         out (pandas.DataFrame) : The same dataframe but + the computed AA properties
#
#     """
#     out = df.copy()
#
#     out[f'{prefix}aliphatic_index'] = out[seq_col].apply(lambda x: peptides.Peptide(x).aliphatic_index())
#     out[f'{prefix}boman'] = out[seq_col].apply(lambda x: peptides.Peptide(x).boman())
#     out[f'{prefix}hydrophobicity'] = out[seq_col].apply(lambda x: peptides.Peptide(x).hydrophobicity())
#     out[f'{prefix}isoelectric_point'] = out[seq_col].apply(lambda x: peptides.Peptide(x).isoelectric_point())
#     # out['PD2'] = out[seq_col].apply(lambda x: peptides.Peptide(x).physical_descriptors()[1])
#     # out['charge_7_4'] = out[seq_col].apply(lambda x: peptides.Peptide(x).charge(pH=7.4))
#     # out['charge_6_65'] = out[seq_col].apply(lambda x: peptides.Peptide(x).charge(pH=6.65))
#     if do_vhse:
#         vhse = out[seq_col].apply(lambda x: peptides.Peptide(x).vhse_scales())
#         # for i in range(1, 9):
#         #     out[f'VHSE{i}'] = [x[i - 1] for x in vhse]
#         for i in [1, 3, 7, 8]:
#             out[f'VHSE{i}'] = [x[i - 1] for x in vhse]
#
#     # Some hardcoded bs
#     return out, ['aliphatic_index', 'boman', 'hydrophobicity',
#                  'isoelectric_point', 'VHSE1', 'VHSE3', 'VHSE7', 'VHSE8']
#

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


# old pfr carlos code

# # Function for encoding the peptide lengths as one-hot
# def pep_len_1hot(df_seq, max_len, window_size, min_length=13, max_length=21):
#     # Define the range of possible sequence lengths
#     seq_lens = df_seq.str.len()
#     # Create an empty NumPy array for one-hot encoding
#     seq_lens_1hot = np.zeros((len(df_seq), (max_length - min_length) + 2), dtype=int)
#
#     # Fill the one-hot array
#     for i, length in enumerate(seq_lens):
#         if length < min_length:
#             seq_lens_1hot[i, 0] = 1  # Group peptides below length 13
#         elif length > max_length:
#             seq_lens_1hot[i, -1] = 1  # Group peptides above length 21
#         else:
#             seq_lens_1hot[i, length - min_length + 1] = 1  # Group peptides in between
#
#     expanded_tensor = torch.from_numpy(seq_lens_1hot).unsqueeze(1).expand(-1, max_len - window_size + 1, -1)
#
#     # print('Peptide lengths encoded for this dataset completed')
#
#     return expanded_tensor

# Function to calculate the mean position values of fixed-size (3) flanking regions of each motif
# def PFR_calculation(sequences, all_xmask, max_len, window_size=9):
#     # Define output
#     data = encode_batch(sequences, max_len, 'BL62FREQ',
#                         None)  # Coding according the FREQ Blosum matrix (better for PFR)
#     # data has shape (len(sequences), max_len, 20)
#
#     all_pfr = torch.empty((data.shape[0], max_len - window_size + 1, 2, 20))  # Initialize the output tensor
#     for j in range(0, data.shape[0], 1):
#
#         seq = data[j, :]
#
#         # Previous PFR mask definition
#         PFR_mask_before = 3 * torch.ones((max_len - window_size + 1, 1))
#         PFR_mask_before[:3, 0] = torch.tensor([0, 1, 2], dtype=torch.float32)  # First three elements to 1 and 2
#
#         # After PFR mask definition (according to their x_mask)
#         PFR_mask_after = torch.clone(all_xmask[j]) * 3
#
#         # Modification of the previous values before 0 (only if zero exists)
#         zero_indices = torch.where(PFR_mask_after == 0)[0]
#         if zero_indices.numel() > 0:  # Check if there are any zero indices
#             zero_index = zero_indices[0]  # Get the first zero index
#             PFR_mask_after[zero_index - 3] = 2 if zero_index >= 3 else PFR_mask_after[zero_index - 3]
#             PFR_mask_after[zero_index - 2] = 1 if zero_index >= 2 else PFR_mask_after[zero_index - 2]
#             PFR_mask_after[zero_index - 1] = 0 if zero_index >= 1 else PFR_mask_after[zero_index - 1]
#
#         for i in range(0, seq.shape[0] - window_size + 1, 1):
#             # Define the WS-mer
#             peptide = seq[i:i + window_size]
#
#             # Store the resulting PFR tensors
#             prev_pfr = torch.sum(seq[i - int(PFR_mask_before[i]):i], dim=0, keepdim=True) / 3
#             after_pfr = torch.sum(seq[i + window_size:i + window_size + int(PFR_mask_after[i])], dim=0,
#                                   keepdim=True) / 3
#
#             # Store the PFR tensors in the output tensor
#             all_pfr[j, i, 0] = prev_pfr
#             all_pfr[j, i, 1] = after_pfr
#
#     return all_pfr.flatten(start_dim=2)
#
# # Function to calculate the length of the flanking regions of each motif
# def FR_lengths(all_xmask, max_len, window_size=9):
#     # Define output
#     all_FR_len = torch.empty((all_xmask.shape[0], max_len - window_size + 1, 2, 2))
#     # Number of total windows for each sequence
#     len_masks = all_xmask.shape[1]
#
#     for j in range(0, all_xmask.shape[0], 1):
#         # After FR mask
#         FR_mask_after = np.array(all_xmask[j]).reshape(-1)
#         # Count of non-zero values
#         count_nonzero = np.count_nonzero(FR_mask_after) - 1
#         # Filling with 0 according to the mask
#         FR_len_after = np.concatenate([np.arange(count_nonzero, 0, -1), np.zeros(len_masks - count_nonzero)])
#
#         # Before FR mask is always the same
#         FR_len_before = np.arange(max_len - window_size + 1)
#
#         # Transformation of length arrays
#         FR_len_before = FR_len_before / (FR_len_before + 1)
#         FR_len_after = FR_len_after / (FR_len_after + 1)
#
#         # Store the LEN_FR arrays as tensors in the output tensor
#         all_FR_len[j, :, 0, 0] = (torch.tensor(FR_len_before))
#         all_FR_len[j, :, 0, 1] = (torch.tensor(1 - FR_len_before))
#         all_FR_len[j, :, 1, 0] = (torch.tensor(FR_len_after))
#         all_FR_len[j, :, 1, 1] = (torch.tensor(1 - FR_len_after))
#
#     return all_FR_len.flatten(start_dim=2)
#
