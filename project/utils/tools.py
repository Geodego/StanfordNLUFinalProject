from pathlib import Path
import sys
import os
import torch
from torch import nn
import time

ROOT_DIRECTORY = 'StanfordNLUFinalProject'
PROJECT_DIRECTORY = 'project'


def get_root_path():
    root_path = Path(sys.path[0].split(ROOT_DIRECTORY)[0] + ROOT_DIRECTORY)
    return root_path


def get_project_directory_path():
    root_path = get_root_path()
    project_path = os.path.join(root_path, PROJECT_DIRECTORY)
    return project_path


def get_directory_path(directory: str):
    project_path = get_project_directory_path()
    # list of project subdirectories
    project_sub = [x for x in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, x))]
    if directory in project_sub:
        path = os.path.join(project_path, directory)
        return path
    else:
        raise Exception('{} is not a proper directory'.format(directory))


def save_model(model: nn.Module, file_name: str):
    """
    Save a model state_dict in project/data/pretrained_models
    :param model:
    :param file_name: name of the file where the states will be saved with no extension. Should start with the name of
    the class of the model followed by _. ex(EncoderDecoder_withglove)
    :return:
    """
    data_path = get_directory_path('data')
    file_path = os.path.join(data_path, 'pretrained_models/' + file_name + '.pt')
    torch.save(model.state_dict(), file_path)


def load_model_states(model: nn.Module, file_name: str):
    """
    load trained parameters of model, saved in file_name
    :param model:
    :param file_name: name of the file where the states are saved with no extension.
    :return:
    pretrained model
    """
    data_path = get_directory_path('data')
    file_path = os.path.join(data_path, 'pretrained_models/' + file_name + '.pt')
    model.load_state_dict(torch.load(file_path))
    return model


def select_data_with_max_length_sentence(max_length: int, tokens_list: list, *args):
    """
    Select the data within a corpus the data corresponding to sentences with less token than max length.
    ex: select_data_with_max_length_sentence(max_length= 10, sentences= tokens_list, texts, colors)
    would return the data in tokens_list, texts and colors corresponding to tokens in tokens_list with less than 10 items
    :param max_length: criteria applied to select items in sentences
    :param tokens_list: list of tokenized utterances
    :param args: other iterables of same length as tokens_list, corresponding to data linked to the one in tokens_list.
    Could be the sentences that have been tokenized, or corresponding colors...
    :return: tuple of list of iterables
    """
    new_args = [[]]
    if args:
        for _ in args:
            new_args.append([])  # initiate an empty list for each iterable in args
    for item in zip(tokens_list, *args):
        if len(item[0]) <= max_length:
            new_args[0].append(item[0])
            for i, x in enumerate(item[1:]):
                new_args[i + 1].append(x)
    return tuple(new_args)


def get_optimizer(name: str):
    """Return the optimizer corresponding to name"""
    return getattr(torch.optim)


def time_calc(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            length = (te - ts) / 60
            print('execution time of {}  {:.2f} mn'.format(method.__name__, length))
        return result

    return timed


