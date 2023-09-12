import pandas as pd
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath('../..')
if module_path not in sys.path:
    sys.path.append(module_path)
#
# module_path = os.path.abspath(os.path.join('../'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import torch
from src.datasets import get_NNAlign_dataloader

dataset_keys = ['max_len', 'window_size', 'encoding', 'seq_col', 'target_col', 'pad_scale', 'batch_size']

dataset_params = {'max_len': 21, 'window_size': 9, 'encoding': 'BL50LO',
                  'seq_col': 'Sequence', 'target_col': 'target', 'pad_scale': -15, 'batch_size': 64}
torch.manual_seed(1)
train_df = pd.read_csv('/Users/riwa/Documents/code/PyNNalign/data/NetMHCIIpan_train/drb1_0301_f0.csv')
train_loader, train_dataset = get_NNAlign_dataloader(train_df, indel=False, return_dataset=True, **dataset_params)


for x,mask,y in train_loader:
    print(mask[0])
    break
train_dataset.burn_in(False)

for x, mask, y in train_loader:
    print(mask[0])
    break

train_dataset.burn_in(True)
for x, mask, y in train_loader:
    print(mask[0])
    break

