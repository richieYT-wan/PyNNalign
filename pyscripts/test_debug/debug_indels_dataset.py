import os, sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.models import NNAlign
from src.datasets import NNAlignDatasetEFSinglePass
from src.data_processing import AA_KEYS
from src.torch_utils import save_checkpoint, load_checkpoint
import pandas as pd
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler

df = pd.read_csv('../../data/mhc1_el_sub10k/mhc1_el_subsampled.csv')
df['flag'] = df['sequence'].apply(lambda x: any([z not in AA_KEYS for z in x]))
df = df.query('not flag')
dataset = NNAlignDatasetEFSinglePass(df, max_len=13, window_size=9,encoding='BL50LO',
                                     seq_col='sequence', target_col='target',pad_scale=-20, indel=True)

loader = dataset.get_dataloader(200, SequentialSampler)

for data in loader:
    x_tensor = data[0]
    x_mask = data[1]
    y = data[-1]

    break
print('stop')

