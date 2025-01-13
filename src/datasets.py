import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.data_processing import PSEUDOSEQDICT, encode, encode_batch, get_structural_info_for_peptide, encode_with_structural_info
from src.data_processing import get_indel_windows_with_structure, batch_insertion_deletion, batch_indel_mask,\
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
    TODO: CLASS TO PHASE OUT
    """

    #@profile
    def __init__(self, df: pd.DataFrame, max_len: int, window_size: int, encoding: str = 'onehot',
                 seq_col: str = 'sequence', target_col: str = 'target', pad_scale: float = None, indel: bool = False,
                 burnin_alphabet: str = 'ILVMFYW', feature_cols: list = ['placeholder'], add_pseudo_sequence=False,
                 pseudo_seq_col: str = 'pseudoseq', add_pfr=False, add_fr_len=False, add_pep_len=False, min_clip=None,
                 max_clip=None, burn_in=None):
        # start = dt.now()
        super(NNAlignDatasetEFSinglePass, self).__init__()

        # Compile encoded sequences into a batch tensor
        
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
        # TODO : Phase out this class
        if len(feature_cols) > 0:
            self.extra_features_flag = True
        else:
            self.extra_features_flag = False

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
    Updated 10.03.24 : CLASS TO USE HERE
    """

    #@profile
    def __init__(self, df: pd.DataFrame, max_len: int, window_size: int, fasta_data: str = None, structural_data: str = None, encoding: str = 'onehot',
                 seq_col: str = 'sequence', target_col: str = 'target', pad_scale: float = None, indel: bool = False,
                 burnin_alphabet: str = 'ILVMFYW', feature_cols: list = ['placeholder'], add_pseudo_sequence=False,
                 add_pfr=False, add_fr_len=False, add_pep_len=False, min_clip=None, max_clip=None, burn_in=None, add_structure = False, add_mean_structure=False):

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
        if not add_structure:
            matrix_dim = 20
            x = encode_batch(df[seq_col], max_len, encoding, pad_scale)
        if add_structure:
            matrix_dim = 25
            encoded_sequences = []    
            for index, row in df.iterrows():
                peptide_sequence = row[seq_col]
                protein_id = row['protein_id']
                structural_info = get_structural_info_for_peptide(protein_id, peptide_sequence, fasta_data, structural_data)
                encoded_sequence = encode_with_structural_info(peptide_sequence, structural_info, max_len, encoding, pad_scale)
                encoded_sequences.append(encoded_sequence)

            x = torch.stack(encoded_sequences)
        #print(x.shape)
        #print(x)
        # query_time = dt.now()
        y = torch.from_numpy(df[target_col].values).float().view(-1, 1)
        # encode_time = dt.now()    
        # Creating the mask to allow selection of kmers without padding
        len_tensor = torch.from_numpy(df['len'].values)
        #print(len_tensor.shape)
        x_mask = len_tensor - window_size
        #print(x_mask.shape)
        range_tensor = torch.arange(max_len - window_size + 1).unsqueeze(0).repeat(len(x), 1)
        # Mask for Kmers + padding
        x_mask = (range_tensor <= x_mask.unsqueeze(1)).float().unsqueeze(-1)
    

        # Expand the kmers windows for base sequence without indels
        x = x.unfold(1, window_size, 1).transpose(2, 3) \
            .reshape(len(x), max_len - window_size + 1, window_size, matrix_dim)
        # Creating indels window and mask
        if indel:
            if add_structure:
                sequence_groups = []
                for index, row in df.iterrows():
                    peptide_sequence = row[seq_col]
                    protein_id = row['protein_id']
                    # Assume get_structural_info_for_peptide returns the needed structural info
                    structural_info = get_structural_info_for_peptide(protein_id, peptide_sequence, fasta_data, structural_data)
                    
                    # Generate indel windows for the sequence and adjust structural info accordingly
                    indel_windows, adjusted_structural_infos = get_indel_windows_with_structure(peptide_sequence, structural_info, window_size)

                    # Temporary list to store encoded windows for the current sequence
                    temp_encoded_sequences = []
                    for window_sequence, window_structural_info in zip(indel_windows, adjusted_structural_infos):
                        encoded_sequence = encode_with_structural_info(window_sequence, window_structural_info, max_len, encoding, pad_scale)
                        temp_encoded_sequences.append(encoded_sequence)

                    
                    sequence_tensor = torch.stack(temp_encoded_sequences)
                    sequence_groups.append(sequence_tensor)

                
                x_indel = torch.stack(sequence_groups)
            else:
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
            # TODO:
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

        if add_mean_structure:
            struct_cols = ['rsa','pq3_H', 'pq3_E', 'pq3_C', 'disorder']
            x_structs = torch.cat([torch.tensor(df[col].apply(lambda x: np.mean([float(z) for z in x.split(',')])).values).unsqueeze(1) for col in struct_cols], dim=1)
            # Expand and tile to cat on dim 2
            x_structs = x_structs.unsqueeze(1).tile(self.x_tensor.shape[1], 1)
            self.x_tensor = torch.cat([self.x_tensor, x_structs], dim=2)

        self.x_tensor = self.x_tensor.float()
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
                # TODO : Bad implementation ; the extra_features x_feats that is used in the models
                #        is actually just the pseudo sequence ; the other features (pep_len, pfr, fr, are just added to the x_tensor)
                # Do the HLA pseudoseq return on the fly instead of pre-expanding and saving
                return self.x_tensor[idx], self.burn_in_mask[idx], self.pseudoseq_tensormap[self.hla_tag[idx]], self.y[idx]
            else:
                # 2
                return self.x_tensor[idx], self.burn_in_mask[idx], self.y[idx]
        else:
            if self.extra_features_flag:
                # 3
                # Do the HLA pseudoseq return on the fly instead of pre-expanding and saving
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