import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.data_processing import PSEUDOSEQDICT, encode, encode_batch
from src.data_processing import batch_insertion_deletion, batch_indel_mask,\
    get_indel_windows, get_pfr_values, get_fr_lengths, get_pep_len_onehot
from memory_profiler import profile
from datetime import datetime as dt


class SuperDataset(Dataset):
    def __init__(self, x=torch.empty([100, 1])):
        super(SuperDataset, self).__init__()
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

    def get_dataset(self):
        return self

    def get_dataloader(self, batch_size, sampler, **kwargs):
        dataloader = DataLoader(self, batch_size=batch_size, sampler=sampler(self), **kwargs)
        return dataloader



class NNAlignDatasetEFSinglePass(SuperDataset):
    """
    CLASS TO PHASE OUT
    """

    #@profile
    def __init__(self, df: pd.DataFrame, max_len: int, window_size: int, encoding: str = 'onehot',
                 seq_col: str = 'sequence', target_col: str = 'target', pad_scale: float = None, indel: bool = False,
                 burnin_alphabet: str = 'ILVMFYW', feature_cols: list = ['placeholder'], add_pseudo_sequence=False,
                 pseudo_seq_col: str = 'pseudoseq', add_pfr=False, add_fr_len=False, add_pep_len=False, min_clip=None,
                 max_clip=None, burn_in=None):
        # start = dt.now()
        super(NNAlignDatasetEFSinglePass, self).__init__()
        # Encoding stuff
        if feature_cols is None:
            feature_cols = []
        # Filter out sequences longer than max_len
        df['len'] = df[seq_col].apply(len)
        df = df.query('len<=@max_len')
        # Then, if indel is False, filter out sequences shorter than windowsize (ex: 8mers for WS=9)
        if not indel:
            df = df.query('len>=@window_size')
        self.burn_in_flag = False
        matrix_dim = 20
        # query_time = dt.now()
        x = encode_batch(df[seq_col], max_len, encoding, pad_scale)
        y = torch.from_numpy(df[target_col].values).float().view(-1, 1)
        # encode_time = dt.now()
        # Creating the mask to allow selection of kmers without padding
        len_tensor = torch.from_numpy(df['len'].values)
        x_mask = len_tensor - window_size
        range_tensor = torch.arange(max_len - window_size + 1).unsqueeze(0).repeat(len(x), 1)
        # Mask for Kmers + padding
        x_mask = (range_tensor <= x_mask.unsqueeze(1)).float().unsqueeze(-1)
        # Expand the kmers windows for base sequence without indels
        x = x.unfold(1, window_size, 1).transpose(2, 3) \
            .reshape(len(x), max_len - window_size + 1, window_size, matrix_dim)
        # Creating indels window and mask
        if indel:
            x_indel = batch_insertion_deletion(df[seq_col], max_len, encoding, pad_scale, window_size)
            # remove padding from indel windows
            x_indel = x_indel[:, :, :window_size, :]
            indel_mask = batch_indel_mask(len_tensor, window_size)
            x = torch.cat([x, x_indel], dim=1)
            x_mask = torch.cat([x_mask, indel_mask], dim=1)

        if burn_in is not None:
            # Creating another mask for the burn-in period+bool flag switch
            self.burn_in_mask = _get_burnin_mask_batch(df[seq_col].values, max_len, window_size,
                                                       burnin_alphabet).unsqueeze(-1)
            if indel:
                indel_burn_in_mask = _get_indel_burnin_mask_batch(df[seq_col].values, window_size,
                                                                  burnin_alphabet).unsqueeze(-1)
                self.burn_in_mask = torch.cat([self.burn_in_mask, indel_burn_in_mask], dim=1)

        # Expand and unfold the sub kmers and the target to match the shape ; contiguous to allow for view operations
        self.x_tensor = x.flatten(2, 3).contiguous()
        self.x_mask = x_mask

        # kmer_time = dt.now()
        self.y = y.contiguous()
        self.x_features = torch.empty((len(x),))
        # Add extra features
        if len(feature_cols) > 0:
            # TODO: When you add more features you need to concatenate to x_pseudosequence and save it to self.x_features
            # these are NUMERICAL FEATURES like %Rank, expression, etc. of shape (N, len(feature_cols))
            # x_features = torch.from_numpy(df[feature_cols].values).float()

            self.extra_features_flag = True
        else:
            self.extra_features_flag = False

        #  TODO dictmap for 9mer look-up and see if how many duplicated and can we save memory
        #
        if add_pseudo_sequence:
            x_pseudoseq = encode_batch(df[pseudo_seq_col], 34, encoding, pad_scale)
            x_pseudoseq = x_pseudoseq.flatten(start_dim=1)
            self.x_features = x_pseudoseq
            self.extra_features_flag = True
            # ps_time = dt.now()
        if add_pfr:
            x_pfr = get_pfr_values(df[seq_col].values, max_len, window_size, indel=indel)
            self.x_tensor = torch.cat([self.x_tensor, x_pfr], dim=2)
            # pfr_time = dt.now()
        if add_fr_len:
            x_fr_len = get_fr_lengths(len_tensor, max_len, window_size, indel=indel)
            self.x_tensor = torch.cat([self.x_tensor, x_fr_len], dim=2)
            # pfr_len_time = dt.now()
        if add_pep_len:
            min_clip = len_tensor.min().item() if min_clip is None else min_clip
            max_clip = len_tensor.max().item() if max_clip is None else max_clip
            x_pep_len = get_pep_len_onehot(len_tensor, max_len, window_size, min_clip, max_clip, indel=indel)
            self.x_tensor = torch.cat([self.x_tensor, x_pep_len], dim=2)
            # peplen_time = dt.now()

        # Saving df in case it's needed
        self.df = df
        self.len = len(x)
        self.max_len = max_len
        self.seq_col = seq_col
        self.window_size = window_size
        # elapsed_query = query_time - start
        # elapsed_encode = encode_time - start
        # elapsed_kmer = kmer_time - start
        # elapsed_ps = ps_time - start
        # elapsed_pfr = pfr_time - start
        # elapsed_pfrlen = pfr_len_time - start
        # elapsed_peplen = peplen_time - start
        # elapsed_query = divmod(elapsed_query.seconds, 60)
        # print('elapsed_query', f'{elapsed_query[0]} minutes {elapsed_query[1]} secs')
        # elapsed_encode = divmod(elapsed_encode.seconds, 60)
        # print('elapsed_encode', f'{elapsed_encode[0]} minutes {elapsed_encode[1]} secs')
        # elapsed_kmer = divmod(elapsed_kmer.seconds, 60)
        # print('elapsed_kmer', f'{elapsed_kmer[0]} minutes {elapsed_kmer[1]} secs')
        # elapsed_ps = divmod(elapsed_ps.seconds, 60)
        # print('elapsed_ps', f'{elapsed_ps[0]} minutes {elapsed_ps[1]} secs')
        # elapsed_pfr = divmod(elapsed_pfr.seconds, 60)
        # print('elapsed_pfr', f'{elapsed_pfr[0]} minutes {elapsed_pfr[1]} secs')
        # elapsed_pfrlen = divmod(elapsed_pfrlen.seconds, 60)
        # print('elapsed_pfrlen', f'{elapsed_pfrlen[0]} minutes {elapsed_pfrlen[1]} secs')
        # elapsed_peplen = divmod(elapsed_peplen.seconds, 60)
        # print('elapsed_peplen', f'{elapsed_peplen[0]} minutes {elapsed_peplen[1]} secs')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """ Returns the appropriate input tensors (X, ..., y) depending on the bool flags
        A bit convoluted return, but basically 4 conditions:
            1. No burn-in, no extra features --> returns the normal x_tensor, kmers mask, target
            2. Burn-in, no extra features --> returns the normal x_tensor, burn-in mask, target
            3. No Burn-in, + extra features --> returns the normal x_tensor, kmers mask, x_features, target
            4. Burn-in, + extra features --> returns the normal x_tensor, burn-in mask, x_features, target
        :param idx:
        :return:
        """
        if self.burn_in_flag:
            if self.extra_features_flag:
                # 4
                # print(f'Tensor, Burn_in_mask, x_features, and y shapes: {self.x_tensor[idx].shape}, {self.burn_in_mask[idx].shape}, {self.x_features[idx].shape}, {self.y[idx].shape}')

                return self.x_tensor[idx], self.burn_in_mask[idx], self.x_features[idx], self.y[idx]
            else:
                # 2
                return self.x_tensor[idx], self.burn_in_mask[idx], self.y[idx]
        else:
            if self.extra_features_flag:
                # 3
                return self.x_tensor[idx], self.x_mask[idx], self.x_features[idx], self.y[idx]
            else:
                # 1
                return self.x_tensor[idx], self.x_mask[idx], self.y[idx]

    def burn_in(self, flag):
        self.burn_in_flag = flag


class NNAlignDataset(SuperDataset):
    """
    Class to test PSEUDOSEQ on the fly
    """

    #@profile
    def __init__(self, df: pd.DataFrame, max_len: int, window_size: int, encoding: str = 'onehot',
                 seq_col: str = 'sequence', target_col: str = 'target', pad_scale: float = None, indel: bool = False,
                 burnin_alphabet: str = 'ILVMFYW', feature_cols: list = ['placeholder'], add_pseudo_sequence=False,
                 add_pfr=False, add_fr_len=False, add_pep_len=False, min_clip=None, max_clip=None, burn_in=None):
        # start = dt.now()
        super(NNAlignDataset, self).__init__()
        # Encoding stuff
        if feature_cols is None:
            feature_cols = []
        # Filter out sequences longer than max_len
        if 'len' not in df.columns:
            df['len'] = df[seq_col].apply(len)
        df = df.query('len<=@max_len')
        # Then, if indel is False, filter out sequences shorter than windowsize (ex: 8mers for WS=9)
        if not indel:
            df = df.query('len>=@window_size')

        matrix_dim = 20
        # query_time = dt.now()
        x = encode_batch(df[seq_col], max_len, encoding, pad_scale)
        y = torch.from_numpy(df[target_col].values).float().view(-1, 1)
        # encode_time = dt.now()
        # Creating the mask to allow selection of kmers without padding
        len_tensor = torch.from_numpy(df['len'].values)
        x_mask = len_tensor - window_size
        range_tensor = torch.arange(max_len - window_size + 1).unsqueeze(0).repeat(len(x), 1)
        # Mask for Kmers + padding
        x_mask = (range_tensor <= x_mask.unsqueeze(1)).float().unsqueeze(-1)
        # Expand the kmers windows for base sequence without indels
        x = x.unfold(1, window_size, 1).transpose(2, 3) \
            .reshape(len(x), max_len - window_size + 1, window_size, matrix_dim)
        # Creating indels window and mask
        if indel:
            x_indel = batch_insertion_deletion(df[seq_col], max_len, encoding, pad_scale, window_size)
            # remove padding from indel windows
            x_indel = x_indel[:, :, :window_size, :]
            indel_mask = batch_indel_mask(len_tensor, window_size)
            x = torch.cat([x, x_indel], dim=1)
            x_mask = torch.cat([x_mask, indel_mask], dim=1)

        # Creating another mask for the burn-in period+bool flag switch
        if burn_in is not None:
            # Creating another mask for the burn-in period+bool flag switch
            self.burn_in_mask = _get_burnin_mask_batch(df[seq_col].values, max_len, window_size,
                                                       burnin_alphabet).unsqueeze(-1)
            if indel:
                indel_burn_in_mask = _get_indel_burnin_mask_batch(df[seq_col].values, window_size,
                                                                  burnin_alphabet).unsqueeze(-1)
                self.burn_in_mask = torch.cat([self.burn_in_mask, indel_burn_in_mask], dim=1)

        self.burn_in_flag = False

        # Expand and unfold the sub kmers and the target to match the shape ; contiguous to allow for view operations
        self.x_tensor = x.flatten(2, 3).contiguous()
        self.x_mask = x_mask

        # kmer_time = dt.now()
        self.y = y.contiguous()
        self.x_features = torch.empty((len(x),))
        # Add extra features
        if len(feature_cols) > 0:
            # TODO: When you add more features you need to concatenate to x_pseudosequence and save it to self.x_features
            # these are NUMERICAL FEATURES like %Rank, expression, etc. of shape (N, len(feature_cols))
            # x_features = torch.from_numpy(df[feature_cols].values).float()
            self.extra_features_flag = True
        else:
            self.extra_features_flag = False

        if add_pseudo_sequence:
            # Use a dictionary to encode on the fly
            self.pseudoseq_tensormap = {k: encode(v, 34, encoding, pad_scale).flatten(start_dim=0) for k, v
                                        in
                                        PSEUDOSEQDICT.items()}
            self.hla_tag = df['HLA'].values
            self.extra_features_flag = True
            # ps_time = dt.now()
        if add_pfr:
            # x_pfr = PFR_calculation(df[seq_col], self.x_mask, max_len, window_size)
            x_pfr = get_pfr_values(df[seq_col].values, max_len, window_size, indel=indel)
            self.x_tensor = torch.cat([self.x_tensor, x_pfr], dim=2)
            # pfr_time = dt.now()
        if add_fr_len:
            x_fr_len = get_fr_lengths(len_tensor, max_len, window_size, indel=indel)
            self.x_tensor = torch.cat([self.x_tensor, x_fr_len], dim=2)
            # pfr_len_time = dt.now()
        if add_pep_len:
            min_clip = len_tensor.min().item() if min_clip is None else min_clip
            max_clip = len_tensor.max().item() if max_clip is None else max_clip
            x_pep_len = get_pep_len_onehot(len_tensor, max_len, window_size, min_clip, max_clip, indel=indel)
            self.x_tensor = torch.cat([self.x_tensor, x_pep_len], dim=2)
            # peplen_time = dt.now()

        # Saving df in case it's needed
        self.df = df
        self.len = len(self.x_tensor)
        self.max_len = max_len
        self.seq_col = seq_col
        self.window_size = window_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """ Returns the appropriate input tensors (X, ..., y) depending on the bool flags
        A bit convoluted return, but basically 4 conditions:
            1. No burn-in, no extra features --> returns the normal x_tensor, kmers mask, target
            2. Burn-in, no extra features --> returns the normal x_tensor, burn-in mask, target
            3. No Burn-in, + extra features --> returns the normal x_tensor, kmers mask, x_features, target
            4. Burn-in, + extra features --> returns the normal x_tensor, burn-in mask, x_features, target
        :param idx:
        :return:
        """
        if self.burn_in_flag:
            if self.extra_features_flag:
                # Do the HLA pseudoseq return on the fly instead of pre-expanding and saving
                x_pseudoseq = self.pseudoseq_tensormap[self.hla_tag[idx]]
                return self.x_tensor[idx], self.burn_in_mask[idx], self.pseudoseq_tensormap[self.hla_tag[idx]], self.y[idx]
            else:
                # 2
                return self.x_tensor[idx], self.burn_in_mask[idx], self.y[idx]
        else:
            if self.extra_features_flag:
                # 3
                # Do the HLA pseudoseq return on the fly instead of pre-expanding and saving
                x_pseudoseq = self.pseudoseq_tensormap[self.hla_tag[idx]]
                return self.x_tensor[idx], self.x_mask[idx], self.pseudoseq_tensormap[self.hla_tag[idx]], self.y[idx]
            else:
                # 1
                return self.x_tensor[idx], self.x_mask[idx], self.y[idx]

    def burn_in(self, flag):
        self.burn_in_flag = flag


def _get_burnin_mask_batch(sequences, max_len, window_size, alphabet='ILVMFYW'):
    return torch.stack([_get_burnin_mask(x, max_len, window_size, alphabet) for x in sequences])


def _get_burnin_mask(seq, max_len, window_size, alphabet='ILVMFYW'):
    mask = torch.tensor([x in alphabet for i, x in enumerate(seq) if i < len(seq) - window_size + 1]).float()
    return F.pad(mask, (0, (max_len - window_size + 1) - len(mask)), 'constant', 0)


def _get_indel_burnin_mask(seq, window_size, alphabet='ILVMFYW'):
    indel_windows = get_indel_windows(seq, window_size)
    return torch.tensor([x[0] in alphabet for x in indel_windows]).float()


def _get_indel_burnin_mask_batch(sequences, window_size, alphabet='ILVMFYW'):
    return torch.stack([_get_indel_burnin_mask(x, window_size, alphabet) for x in sequences])


# Stupid shit for memory profiler

class UglyWorkAround(SuperDataset):

    def __init__(self, df: pd.DataFrame, max_len: int, window_size: int, encoding: str = 'onehot',
                 seq_col: str = 'sequence', target_col: str = 'target', pad_scale: float = None, indel: bool = False,
                 burnin_alphabet: str = 'ILVMFYW', feature_cols: list = ['placeholder'], add_pseudo_sequence=False,
                 pseudo_seq_col: str = 'pseudoseq', add_pfr=False, add_fr_len=False, add_pep_len=False, add_z=True,
                 burn_in=None):
        super(UglyWorkAround, self).__init__()


# OLD DEPRECATED CODE
# @profile
def get_NNAlign_dataloaderEFSinglePass(df: pd.DataFrame, max_len: int, window_size: int, encoding: str = 'onehot',
                                       seq_col: str = 'Peptide', target_col: str = 'agg_label', pad_scale: float = None,
                                       indel: bool = False, burnin_alphabet: str = 'ILVMFYW', feature_cols: list = None,
                                       batch_size=64, sampler=torch.utils.data.RandomSampler, return_dataset=True,
                                       add_pseudo_sequence=False, pseudo_seq_col: str = 'pseudoseq', add_pfr=False,
                                       add_fr_len=False, add_pep_len=False):
    dataset = NNAlignDatasetEFSinglePass(df, max_len, window_size, encoding, seq_col, target_col, pad_scale, indel,
                                         burnin_alphabet, feature_cols, add_pseudo_sequence, pseudo_seq_col, add_pfr,
                                         add_fr_len, add_pep_len)
    # TODO NEW COLLATE FN ON THE FLY FOR KMERS AND MHC
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler(dataset))
    if return_dataset:
        return dataloader, dataset
    else:
        return dataloader
