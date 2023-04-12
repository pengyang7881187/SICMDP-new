import matplotlib.pyplot as plt
from typing import Tuple, Callable
from utils.utility import update_env_config
from configs import DEFAULT_POLLUTION_CONFIG
from envs import PollutionEnv


def get_monomial_func(p: float) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    assert p >= 1
    def f(x: float) -> float:
        return x ** p

    def f_deri(x: float) -> float:
        return p * (x ** (p - 1.))

    return f, f_deri


def get_lp_func(p: float) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    assert p >= 1
    def f(x: float) -> float:
        return 1. - (1. - abs(x) ** p) ** (1. / p)

    def f_deri(x: float) -> float:
        return (abs(x) ** (p - 1.)) * ((1. - abs(x) ** p) ** (1./p - 1.))

    return f, f_deri


if __name__ == '__main__':
    env_config = DEFAULT_POLLUTION_CONFIG.copy()
    update_env_config(env_config=env_config, key='pollution_func_parameter', val=20.)
    update_env_config(env_config=env_config, key='protect_func_parameter', val=20.)
    update_env_config(env_config=env_config, key='protect_func_multiplier', val=0.005)
    update_env_config(env_config=env_config, key='protect_func_intercept', val=0.02)

    env = PollutionEnv(env_config=env_config)

    predefined_path = get_monomial_func(4)
    straight_path = get_monomial_func(1)
    invalid_path = get_monomial_func(2)

    fineness = 100
    gamma = 1.
    plt.figure()
    discrete_predefined_path = env.discretize_path(*predefined_path)
    discrete_predefined_path_length = discrete_predefined_path.shape[0]
    print(f'Predefined path length: {discrete_predefined_path_length}')
    plt.plot(discrete_predefined_path[:, 0], discrete_predefined_path[:, 1], marker='o', mec='none', ms=4, lw=1, label='predefined')

    discrete_straight_path = env.discretize_path(*straight_path)
    discrete_straight_path_length = discrete_straight_path.shape[0]
    print(f'Straight path length: {discrete_straight_path_length}')
    plt.plot(discrete_straight_path[:, 0], discrete_straight_path[:, 1], marker='o', mec='none', ms=4, lw=1, label='straight')

    discrete_invalid_path = env.discretize_path(*invalid_path)
    discrete_invalid_path_length = discrete_invalid_path.shape[0]
    print(f'Invalid path length: {discrete_invalid_path_length}')
    plt.plot(discrete_invalid_path[:, 0], discrete_invalid_path[:, 1], marker='o', mec='none', ms=4, lw=1, label='invalid')
    plt.legend()
    plt.show()


    print('Predefined path:')
    env.evaluate_along_discrete_path(discrete_path=discrete_predefined_path, fineness=fineness, gamma=gamma, name='Predefined')

    print('Straight path:')
    env.evaluate_along_discrete_path(discrete_path=discrete_straight_path, fineness=fineness, gamma=gamma, name='Straight')

    print('Invalid path:')
    env.evaluate_along_discrete_path(discrete_path=discrete_invalid_path, fineness=fineness, gamma=gamma, name='Invalid')










