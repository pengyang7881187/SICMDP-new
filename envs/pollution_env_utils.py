import math
import torch
import numpy as np

from torch import Tensor
from numpy import ndarray
from functools import partial
from typing import Callable, Tuple

from utils import ArrayType


# Candidate pollute functions.
def pollute_reciprocal_alpha_func(alpha: float, d: ArrayType | float) -> ArrayType | float:
    # d is the distant.
    assert alpha > 0
    return 1. / (1. + d ** alpha)


def pollute_reciprocal_alpha_grad_d(alpha: float, d: ArrayType | float) -> ArrayType | float:
    # d is the distant.
    return - alpha * (d ** (alpha - 1.)) / ((1. + d ** alpha) ** 2.)


def pollute_reciprocal_alpha_hess_d(alpha: float, d: ArrayType | float) -> ArrayType | float:
    # d is the distant.
    return (2. * (alpha ** 2.) * (d ** (2. * alpha - 2.))) / ((d ** alpha + 1.) ** 3.)\
        - ((alpha - 1.) * alpha * (d ** (alpha - 2.)))/((d ** alpha + 1.) ** 2.)


def pollute_exponential_beta_func(beta: float, d: ArrayType | float) -> ArrayType | float:
    # d is the distant.
    assert beta > 0
    if isinstance(d, float) or isinstance(d, ndarray):
        return np.exp(- beta * d)
    elif isinstance(d, Tensor):
        return torch.exp(- beta * d)
    else:
        import jax.numpy as jnp
        return jnp.exp(- beta * d)


def pollute_exponential_beta_grad_d(beta: float, d: ArrayType | float) -> ArrayType | float:
    # d is the distant.
    if isinstance(d, float) or isinstance(d, ndarray):
        return - beta * np.exp(- beta * d)
    elif isinstance(d, Tensor):
        return - beta * torch.exp(- beta * d)
    else:
        import jax.numpy as jnp
        return - beta * jnp.exp(- beta * d)


def pollute_exponential_beta_hess_d(beta: float, d: ArrayType | float) -> ArrayType | float:
    # d is the distant.
    if isinstance(d, float) or isinstance(d, ndarray):
        return beta * beta * np.exp(- beta * d)
    elif isinstance(d, Tensor):
        return beta * beta * torch.exp(- beta * d)
    else:
        import jax.numpy as jnp
        return beta * beta * jnp.exp(- beta * d)


def get_pollute_func_grad_and_hess(func_name: str, parameter: float) -> \
        Tuple[
            Callable[[ArrayType | float], ArrayType | float],
            Callable[[ArrayType | float], ArrayType | float],
            Callable[[ArrayType | float], ArrayType | float]
        ]:
    pollute_func_name_dict = {
        'reciprocal': pollute_reciprocal_alpha_func,
        'exponential': pollute_exponential_beta_func
    }
    pollute_grad_name_dict = {
        'reciprocal': pollute_reciprocal_alpha_grad_d,
        'exponential': pollute_exponential_beta_grad_d
    }
    pollute_hess_name_dict = {
        'reciprocal': pollute_reciprocal_alpha_hess_d,
        'exponential': pollute_exponential_beta_hess_d
    }
    func = partial(pollute_func_name_dict[func_name], parameter)
    grad = partial(pollute_grad_name_dict[func_name], parameter)
    hess = partial(pollute_hess_name_dict[func_name], parameter)
    return func, grad, hess


# Candidate protect functions.
def protect_monomial_alpha_func(alpha: float, multiplier: float, intercept: float, d: ArrayType | float) \
        -> ArrayType | float:
    # d is the distant.
    assert alpha > 0 and intercept > 0 and multiplier > 0
    return multiplier * (d ** alpha) + intercept


def protect_monomial_alpha_grad_d(alpha: float, multiplier: float, d: ArrayType | float) -> ArrayType | float:
    # d is the distant.
    if math.isclose(alpha, 0.):
        return 0.
    else:
        return multiplier * alpha * (d ** (alpha - 1.))


def protect_monomial_alpha_hess_d(alpha: float, multiplier: float, d: ArrayType | float) -> ArrayType | float:
    # d is the distant.
    if math.isclose(alpha, 0.) or math.isclose(alpha, 1.):
        return 0.
    else:
        return multiplier * alpha * (alpha - 1.) * (d ** (alpha - 2.))


def protect_exponential_beta_func(beta: float, multiplier: float, intercept: float, d: ArrayType | float) \
        -> ArrayType | float:
    # d is the distant.
    assert beta > 0 and intercept > 0 and multiplier > 0
    if isinstance(d, float) or isinstance(d, ndarray):
        return multiplier * np.exp(beta * d) + (intercept - multiplier)
    elif isinstance(d, Tensor):
        return multiplier * torch.exp(beta * d) + (intercept - multiplier)
    else:
        import jax.numpy as jnp
        return multiplier * jnp.exp(beta * d) + (intercept - multiplier)


def protect_exponential_beta_grad_d(beta: float, multiplier: float, d: ArrayType | float) -> ArrayType | float:
    # d is the distant.
    if isinstance(d, float) or isinstance(d, ndarray):
        return multiplier * beta * np.exp(beta * d)
    elif isinstance(d, Tensor):
        return multiplier * beta * torch.exp(beta * d)
    else:
        import jax.numpy as jnp
        return multiplier * beta * jnp.exp(beta * d)


def protect_exponential_beta_hess_d(beta: float, multiplier: float, d: ArrayType | float) -> ArrayType | float:
    # d is the distant.
    if isinstance(d, float) or isinstance(d, ndarray):
        return multiplier * beta * beta * np.exp(beta * d)
    elif isinstance(d, Tensor):
        return multiplier * beta * beta * torch.exp(beta * d)
    else:
        import jax.numpy as jnp
        return multiplier * beta * beta * jnp.exp(beta * d)


def get_protect_func_grad_and_hess(func_name: str, parameter: float, multiplier: float, intercept: float) ->\
        Tuple[
            Callable[[ArrayType | float], ArrayType | float],
            Callable[[ArrayType | float], ArrayType | float],
            Callable[[ArrayType | float], ArrayType | float]
        ]:
    try:
        # func_type_name should not contain '_'.
        protect_func_name_dict = {
            'monomial': protect_monomial_alpha_func,
            'exponential': protect_exponential_beta_func
        }
        protect_grad_name_dict = {
            'monomial': protect_monomial_alpha_grad_d,
            'exponential': protect_exponential_beta_grad_d
        }
        protect_hess_name_dict = {
            'monomial': protect_monomial_alpha_hess_d,
            'exponential': protect_exponential_beta_hess_d
        }
        func = partial(protect_func_name_dict[func_name], parameter, multiplier, intercept)
        grad = partial(protect_grad_name_dict[func_name], parameter, multiplier)
        hess = partial(protect_hess_name_dict[func_name], parameter, multiplier)
        return func, grad, hess
    except:
        raise RuntimeError('Invalid function name!')
