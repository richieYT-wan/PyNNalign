import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.models import NNAlignEF, NNAlign
from src.datasets import get_NNAlign_dataloader, NNAlignDataset
from src.train_eval import train_model_step, eval_model_step, predict_model
from src.torch_utils import save_checkpoint, load_checkpoint
import pandas as pd
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
import torch.nn.functional as F
from torch import optim
cedar_aligned = pd.read_csv('../data/aligned_icore/230530_cedar_aligned.csv')
n_hidden=25
window_size=5
dropout=0.15
batchnorm=True
feature_cols = ['EL_rank_mut', 'icore_selfsimilarity']
n_extrafeatures= len(feature_cols)
activation = nn.ReLU()
indel, standardize = False, True

# With extra col
train_loader, train_dataset = get_NNAlign_dataloader(cedar_aligned.query('fold!=0 and fold != 1'), max_len=12, window_size=5,
                                                     seq_col='mutant', target_col='target', sampler=RandomSampler,
                                                     feature_cols = feature_cols, return_dataset=True)
valid_loader, valid_dataset = get_NNAlign_dataloader(cedar_aligned.query('fold==0'), max_len=12, window_size=5,
                                                     seq_col='mutant', target_col='target', sampler=SequentialSampler,
                                                     feature_cols = feature_cols, return_dataset=True)
test_loader, test_dataset = get_NNAlign_dataloader(cedar_aligned.query('fold==1'), max_len=12, window_size=5,
                                                   seq_col='mutant', target_col='target', sampler=SequentialSampler,
                                                     feature_cols = feature_cols, return_dataset=True)

model = NNAlignEF(n_hidden, window_size, activation, batchnorm, dropout, indel, standardize,
                  n_extrafeatures, n_hidden, activation, batchnorm, dropout)

model.fit_standardizer(x_tensor=train_dataset.x_tensor, x_mask=train_dataset.x_mask, x_features=train_dataset.x_features)

criterion = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2)
train_stuff = train_model_step(model, criterion, optimizer, train_loader)
valid_stuff = eval_model_step(model, criterion, valid_loader)
save_checkpoint(model, filename='test.pt', dir_path = '../output/debug_extrafeats/')
load_checkpoint(model, filename='test.pt', dir_path = '../output/debug_extrafeats/')
predict_stuff = predict_model(model, test_dataset, test_loader)
