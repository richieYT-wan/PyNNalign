import argparse
import os
import pickle
import pandas as pd
from IPython.display import display_html
from itertools import chain, cycle
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from sklearn.model_selection import KFold
import secrets
import string
from datetime import datetime as dt


def epoch_counter(model, criterion):
    if hasattr(model, 'counter') and hasattr(model, 'increment_counter'):
        model.increment_counter()
    if hasattr(criterion, 'counter') and hasattr(criterion, 'increment_counter'):
        criterion.increment_counter()


def get_class_initcode_keys(class_: object, dict_kwargs: dict) -> list:
    init_code = class_.__init__.__code__
    init_code = class_.__init__.__code__.co_varnames[1:init_code.co_argcount]
    return [x for x in dict_kwargs.keys() if x in init_code]


def get_motif(row, seq_col, window_size):
    # TODO: Pablo, here, at some point you will need to modify this function when insertions/deletions work
    #       Currently, to get the motif we simply read the sequence and use the "best index" from the window
    #       to get the corresponding binding core.
    #       When we do In/Del, we will have extra core indices (as returned by x_mask), and we will need to adapt
    #       to find which binding motif this corresponds to

    return row[seq_col][int(row['core_start_index']):int(row['core_start_index']) + window_size]


def plot_loss_aucs(train_losses, valid_losses, train_aucs, valid_aucs,
                   filename, outdir, dpi=300):
    f, a = plt.subplots(2, 1, figsize=(12, 10))
    a = a.ravel()
    a[0].plot(train_losses, label='train_losses')
    a[0].plot(valid_losses, label='valid_losses')
    a[0].legend()
    a[0].set_title('loss')
    a[1].plot(train_aucs, label='train_aucs')
    a[1].plot(valid_aucs, label='valid_aucs')
    a[1].legend()
    a[1].set_title('AUC')
    a[1].set_xlabel('epochs')
    f.savefig(f'{outdir}{filename}.png', dpi=dpi, bbox_inches='tight')


def get_datetime_string():
    return dt.now().strftime("%y%m%d_%H%M")


def get_random_id(length=6):
    first_character = ''.join(
        secrets.choice(string.digits) for _ in range(2))  # Generate a random digit for the first character
    remaining_characters = ''.join(
        secrets.choice(string.ascii_letters + string.digits) for _ in
        range(length - 2))  # Generate L-2 random characters
    random_string = first_character + remaining_characters
    return random_string


def make_chunks(iterable, chunk_size):
    k, m = divmod(len(iterable), chunk_size)
    return (iterable[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(chunk_size))


def get_kfolds(df, k, xcol, ycol, shuffle=False, random_state=None):
    """ Splits & assigns the fold numbers
    Args:
        df:
        k:
        shuffle:
        random_state:

    Returns:
        df: df with column fold according to the Kfolds
    """
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    df['fold'] = -1
    for i, (train_idx, test_idx) in enumerate(kf.split(df[xcol].values, df[ycol])):
        df.iloc[test_idx, df.columns.get_loc('fold')] = i
    df.fold = df.fold.astype(int)
    return df


def get_palette(palette, n_colors):
    """ 'stretches' stupid fucking palette to have more contrast"""
    if n_colors == 2:
        pal = sns.color_palette(palette, n_colors=5)
        palette = [pal[0], pal[-1]]
    elif n_colors == 3:
        pal = sns.color_palette(palette, n_colors=5)
        palette = [pal[0], pal[2], pal[-1]]
    else:
        nc = int(n_colors * 2)
        pal = sns.color_palette(palette, n_colors=nc)
        palette = [pal[i] for i in range(1, 1 + int(n_colors * 2), 2)]
    return palette


def convert_hla(hla):
    if not hla.startswith('HLA-'):
        hla = 'HLA-' + hla
    return hla.replace('*', '').replace(':', '')


def add_median_labels(ax, fmt='.1%'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


def flatten_product(container):
    """
    Flattens a product or container into a flat list, useful when product/chaining many conditions
    Looks into each sub-element & recursively calls itself
    Args:
        container:
    Returns:

    """
    for i in container:
        if isinstance(i, list) or isinstance(i, tuple):
            for j in flatten_product(i):
                yield j
        else:
            yield i


def display_side(*args, titles=cycle([''])):
    """
    small util to display pd frames side by side
    """
    html_str = ''
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += f'<h2>{title}</h2>'
        html_str += df.to_html().replace('table', 'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)


def str2bool(v):
    """converts str to bool from argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def pkl_dump(obj, filename, dirname=None):
    if dirname is not None:
        mkdirs(dirname)
        filename = os.path.join(dirname, filename)

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        print(f'{os.path.abspath(filename)} saved.')


def pkl_load(filename, dirname=None):
    if dirname is not None:
        filename = os.path.join(dirname, filename)
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            return obj
    except:
        raise ValueError(f'Unable to load or find {os.path.abspath(os.path.join(dirname, filename))}!')


def flatten_level_columns(df: pd.DataFrame, levels=[0, 1]):
    df.columns = [f'{x.lower()}_{y.lower()}'
                  for x, y in zip(df.columns.get_level_values(levels[0]),
                                  df.columns.get_level_values(levels[1]))]
    return df


def convert_path(path):
    return path.replace('\\', '/')


def get_plot_corr(df, cols, which='spearman', title='', figsize=(13, 12.5), palette='viridis'):
    corr = df[cols].corr(which)
    f, a = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(corr.round(2), center=0, xticklabels=corr.columns, yticklabels=corr.columns,
                cmap=palette, vmax=1, vmin=-1, annot=True, square=True, annot_kws={'weight': 'semibold'})
    a.set_xticklabels(a.get_xticklabels(), rotation=30, fontweight='semibold')
    a.set_yticklabels(a.get_yticklabels(), fontweight='semibold')
