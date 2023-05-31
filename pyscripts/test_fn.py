import pandas as pd
from tqdm.auto import tqdm
import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import mkdirs, get_random_id, get_datetime_string

fn = sys.argv[1]
connector = '' if fn == '' else '_'
unique_filename = f'{fn}{connector}{get_datetime_string()}_{get_random_id(5)}'
checkpoint_filename = f'checkpoint_best_{unique_filename}.pt'
outdir = os.path.join('../output/', unique_filename) + '/'
mkdirs(outdir)

