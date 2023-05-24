import os, sys
import pickle
from .utils import mkdirs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def set_mode(models_dict, mode='eval'):
    """
    QOL function to set all models to train or eval
    Args:
        models_dict (dict): dictionary of list of models
        mode (str): mode.lower() should be either 'eval' or 'train'
    Returns:
        the same models_dict
    """
    assert mode.lower() in ['eval', 'train'], 'Please provide a proper mode' \
                                              f'(either "train" or "eval"). You provided "{mode}"'
    mode_bool = mode.lower() == 'train'
    for key, model_list in models_dict.items():
        for model in model_list:
            model.train(mode_bool)


def set_device(models_list, device):
    """
    QOL fct to set all models in a LIST to the same device.
    Only for list and not Dict because copying takes time and space
    Args:
        models_list:
        device:

    Returns:

    """
    for model in models_list:
        model.to(device)


def save_checkpoint(model, name: str = 'checkpoint.pt', dir_path: str = './'):
    """
    Saves a single torch model, with some sanity checks
    Args:
        model: torch model (i.e. anything that inherits from nn.Module and has a state_dict())
        name: the name itself (ex: CNN_model_t0_v1 for a CNN model from the test fold 0, validation fold 1)
        dir_path: path to the directory where the models should be saved

    Returns:
        nothing.
    """
    # Small sanity checks
    assert hasattr(model, 'state_dict'), f'Object of type {type(model)} has no state_dict and can\'t be saved!'
    # Bad practice but fuck it
    if not os.path.exists(dir_path):
        mkdirs(dir_path)
        print(f'Creating {dir_path}; The provided dir path {dir_path} did not exist!')
    if not name.endswith('.pt'):
        name = name + '.pt'
    savepath = os.path.join(dir_path, name)
    torch.save(model.state_dict(), savepath)
    print(f'Model saved at {savepath}')


def load_checkpoint(model, filename: str, dir_path: str = None):
    """
    Loads a model
    Args:
        model:
        name:
        dir_path:

    Returns:
        model: Loaded model, in eval mode.
    """
    if dir_path is not None:
        filename = os.path.join(dir_path, filename)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model


def create_load_model(constructor, filename: str, dir_path: str = None):
    """If provided with a constructor, loads the state_dict and creates the model object from the state_dict

    Assumes that state_dict has a key "init_params" that is used to initialize the model with the constructor.
    Args:
        constructor:
        filename:
        dir_path:

    Returns:
        model: Loaded model, in eval mode, created from the constructor + the state_dict['init_params']
    """
    if dir_path is not None:
        filename = os.path.join(dir_path, filename)
    state_dict = torch.load(filename)
    assert 'init_params' in state_dict.keys(), f'state_dict does not contain init_params key! It has {state_dict.keys()} instead.'
    model = constructor(**state_dict['init_params'])
    model.load_state_dict(state_dict)
    model.eval()
    return model
