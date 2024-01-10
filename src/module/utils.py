import numpy as np
import torch
from collections.abc import Iterable, Sequence, Mapping
from itertools import repeat


def get_available_gpus():
    """
    Returns a list of available GPU IDs.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
        gpu_names = [torch.cuda.get_device_name(i) for i in gpu_ids]
        return gpu_ids, gpu_names
    else:
        return [], []

    

def ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, Sequence):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, Mapping):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output
