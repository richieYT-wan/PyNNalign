import torch
from torch.utils.data import DataLoader, Dataset
from src.data_processing import encode_batch, encode_batch_weighted


class NNAlignDataset(Dataset):
    """
    Here for now, only get encoding and try to
    """

    def __init__(self, df, max_len, window_size, encoding='onehot', seq_col='Peptide',
                 target_col='agg_label', pad_scale=None, indel=False):

        super(NNAlignDataset, self).__init__()
        # Encoding stuff
        df['len'] = df[seq_col].apply(len)
        # l_start = len(df)
        df = df.query('len<=@max_len')
        # l_end = len(df)
        # print(f'Pruning sequences longer than length={max_len}. \nNseqs Before:\t{l_start}\nNseqs After:\t{l_end}')
        matrix_dim = 21 if indel else 20
        x = encode_batch(df[seq_col], max_len, encoding, pad_scale)
        y = torch.from_numpy(df[target_col].values).float().view(-1,1)

        # Creating the mask to allow selection of kmers without padding
        x_mask = torch.from_numpy(df['len'].values)-window_size
        range_tensor = torch.arange(max_len - window_size + 1).unsqueeze(0).repeat(len(x), 1)
        self.x_mask = (range_tensor <= x_mask.unsqueeze(1)).float().unsqueeze(-1)
        # Expand and unfold the sub kmers and the target to match the shape ; contiguous to allow for view operations
        self.x_tensor = x.unfold(1, window_size, 1).transpose(2, 3) \
                  .reshape(len(x), max_len - window_size + 1, window_size, matrix_dim).flatten(2, 3).contiguous()
        self.y = y.contiguous()
        # Saving df in case it's needed
        self.df = df
        self.len = len(x)
        self.max_len = max_len
        self.window_size = window_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """

        return self.x_tensor[idx], self.x_mask[idx], self.y[idx]


def get_NNAlign_dataloader(df, max_len, window_size, encoding='onehot', seq_col='Peptide',
                           target_col='agg_label', pad_scale=None, indel=False, batch_size=64, sampler=torch.utils.data.RandomSampler,
                           return_dataset=False):
    dataset = NNAlignDataset(df, max_len, window_size, encoding, seq_col, target_col, pad_scale, indel)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler(dataset))
    if return_dataset:
        return dataloader, dataset
    else:
        return dataloader
