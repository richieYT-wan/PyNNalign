import torch
from torch.utils.data import DataLoader, Dataset
from src.data_processing import encode_batch, encode_batch_weighted


class NNAlignDataset(Dataset):
    """
    Here for now, only get encoding and try to
    """

    def __init__(self, df, max_len, window_size, indel=False, encoding='onehot', seq_col='Peptide',
                 target_col='agg_label', pad_scale=None):
        super(NNAlignDataset, self).__init__()
        # Encoding stuff
        df['len'] = df[seq_col].apply(len)
        df = df.query('len<=@max_len')
        matrix_dim = 21 if indel else 20
        x = encode_batch(df[seq_col], max_len, encoding, pad_scale)
        y = torch.from_numpy(df[target_col].values).float().view(-1,1)
        # Expand and unfold the sub kmers and the target to match the shape
        self.x = x.unfold(1, window_size, 1).transpose(2, 3) \
            .reshape(len(x), max_len - window_size + 1, window_size, matrix_dim).flatten(2, 3)
        self.y = y #.expand((len(y), max_len - window_size + 1)).view(-1, max_len - window_size + 1, 1)
        # Saving the normal encoded things in case it's needed
        self.encoded_x = x
        self.encoded_y = y
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

        return self.x[idx], self.y[idx]


def get_NNAlign_dataloader(df, max_len, window_size, indel=False, encoding='onehot', seq_col='Peptide',
                           target_col='agg_label', pad_scale=None, batch_size=64, sampler=torch.utils.data.RandomSampler,
                           return_dataset=False):
    dataset = NNAlignDataset(df, max_len, window_size, indel, encoding, seq_col, target_col, pad_scale)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler(dataset))
    if return_dataset:
        return dataloader, dataset
    else:
        return dataloader
