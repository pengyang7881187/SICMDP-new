import os
import time
import torch
import random
import datetime
import numpy as np
import torch.nn as nn

from torch import Tensor
from numpy import ndarray
from functools import wraps
from typing import Dict, Any
from numpy.random import default_rng

ArrayType = ndarray | Tensor

DEFAULT_POLICY_IDX = 'default_policy'


def update_env_config(env_config: Dict[str, Any], key: str, val: Any):
    if key not in env_config:
        raise RuntimeError(f'Invalid key \'{key}\' for the environment config!')
    env_config[key] = val
    return


def timeit_func(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function: {func.__name__}, Time: {total_time:.6f} seconds')
        return result
    return timeit_wrapper


def printit(func):
    @wraps(func)
    def print_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(result)
        return result
    return print_wrapper


# Current time without blank
def get_datetime_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


# Set seed and return a random generator of numpy
def set_seed_and_get_rng(seed: int) -> np.random._generator.Generator:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return default_rng(seed)


def get_device(gpu: int) -> torch.device:
    if gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return torch.device('cpu')
    else:
        return torch.device(f'cuda:{gpu}')


# Vectorize version of np.random.choice, without checking p_array is a valid prob distribution
# dim(p_array)=2, when axis=1, p_array[k, :] is a prob distribution
def vectorize_choice(p_array: ndarray, rng: np.random._generator.Generator, axis=1) -> ndarray:
    p_array_num = p_array.shape[1-axis]
    r = np.expand_dims(rng.random(p_array_num), axis=axis)  # (p_array_num, 1)
    return (p_array.cumsum(axis=axis) > r).argmax(axis=axis)  # argmax will return the first element larger than r



def torch_vectorize_choice(p_tensor: Tensor, device: torch.device, dim=1) -> Tensor:
    p_tensor_num = p_tensor.shape[1-dim]
    r = torch.rand((p_tensor_num, 1), device=device)  # (p_array_num, 1)
    return torch.argmax((p_tensor.cumsum(dim=dim) > r).float(), dim=dim)  # argmax will return the first element larger than r


def log(content: str, logfile=None, silent_flag=False):
    """
    Prints the provided string, and also logs it if a logfile is passed.
    Parameters
    ----------
    content : str
        String to be printed/logged.
    logfile : str (optional)
        File to log into.
    silent_flag: bool
        Flag indicates whether to print.
    """
    content = f"[{get_datetime_str()}] {content}"
    if not silent_flag:
        print(content)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(content, file=f)
    return


def np_softmax(logit: ndarray) -> ndarray:
    softmax = nn.Softmax(dim=logit.ndim-1)
    return softmax(torch.from_numpy(logit)).numpy()


def torch_softmax(logit: Tensor) -> Tensor:
    softmax = nn.Softmax(dim=logit.ndim - 1)
    return softmax(logit)


# Only work for (S, A) policy logit
# Compute \nabla log \pi_\theta(a|s) for give (s, a) pair.
# Support vectorization
def batch_np_derivative_log_softmax(prob: ndarray, s_array: ndarray, a_array: ndarray) -> ndarray:
    S, A = prob.shape
    batch_size = s_array.size
    batch_array = np.arange(batch_size)

    deri = np.zeros((batch_size, S, A))
    deri[batch_array, s_array, a_array] += 1.
    deri[batch_array, s_array, :] -= prob[s_array, :]
    return deri


# Only work for (S, A) policy logit
# Compute \nabla log \pi_\theta(a|s) for give (s, a) pair.
# Support vectorization
def batch_torch_derivative_log_softmax(prob: Tensor, s_array: Tensor, a_array: Tensor, device: torch.device) -> Tensor:
    S, A = prob.shape
    batch_size = len(s_array)
    batch_array = torch.arange(batch_size)

    deri = torch.zeros((batch_size, S, A), device=device)
    deri[batch_array, s_array, a_array] += 1.
    deri[batch_array, s_array, :] -= prob[s_array, :]
    return deri


def np_project_to_l2_ball(vec: ndarray, W: float) -> ndarray:
    norm_vec = np.linalg.norm(vec)
    if norm_vec <= W:
        return vec
    else:
        return (vec / norm_vec) * W


def torch_project_to_l2_ball(vec: Tensor, W: float) -> Tensor:
    norm_vec = torch.linalg.norm(vec)
    if norm_vec <= W:
        return vec
    else:
        return (vec / norm_vec) * W
