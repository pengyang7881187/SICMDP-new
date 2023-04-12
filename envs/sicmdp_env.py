import warnings

import torch
import numpy as np
import gymnasium as gym

from torch import Tensor
from numpy import ndarray
from numpy.random import default_rng
from ray.rllib.env.env_context import EnvContext

from utils import ArrayType

GRID_NOT_GENERATED_FLAG = -1


class Grid:
    def __init__(self, dim_Y, lb_Y, ub_Y):
        self.dim_Y = dim_Y
        self.lb_Y = lb_Y
        self.ub_Y = ub_Y
        self.fineness = GRID_NOT_GENERATED_FLAG
        self.grid = np.empty((1,))
        self.size = 0

    # Reset the grid.
    def reset(self):
        self.fineness = GRID_NOT_GENERATED_FLAG
        self.grid = np.empty((1,))
        self.size = 0
        return

    # Generate the grid.
    def generate(self, fineness=100) -> ndarray:
        if self.fineness == fineness:
            # The grid has been computed.
            return self.grid
        elif self.fineness != -1:
            self.reset()

        self.fineness = fineness
        self.size = (self.fineness + 1) ** self.dim_Y

        coordinate_lst = []
        index_array = np.indices(list(self.dim_Y * [self.fineness + 1]))
        for i in range(self.dim_Y):
            coordinate_lst.append(index_array[i].reshape(-1) *
                                  ((self.ub_Y[i] - self.lb_Y[i]) / self.fineness) + self.lb_Y[i])
        self.grid = np.stack(list(coordinate_lst), axis=1)
        return self.grid


# We only consider Y is a rectangular in R^d
class SICMDPEnv(gym.Env):
    CONSTRAINTS = 'constraints'
    CONSTRAINTS_VALUE = 'constraints_value'
    CONSTRAINTS_ADVANTAGE = 'constraints_advantage'
    CONSTRAINTS_VALUE_TARGET = 'constraints_value_target'  # Q function.

    MAX_VIOLAT_CONSTRAINTS = 'max_violat_constraints'
    MAX_VIOLAT_CONSTRAINTS_VALUE = 'max_violat_constraints_value'
    MAX_VIOLAT_CONSTRAINTS_ADVANTAGE = 'max_violat_constraints_advantage'
    MAX_VIOLAT_CONSTRAINTS_VALUE_TARGET = 'max_violat_constraints_value_target'  # Q function.
    def __init__(self, env_config: EnvContext):
        self.env_config = env_config  # For print.

        self.name = env_config['name']

        # The seed is only used to control sampling y from Y.
        self.seed = env_config['seed']
        self.rng = default_rng(self.seed)

        # Dimension of Y.
        self.dim_Y = env_config['dim_Y']

        # Range of rectangular Y.
        self.lb_Y = env_config['lb_Y']
        self.ub_Y = env_config['ub_Y']
        # Used in bound argument in scipy.optimize.minimize.
        self.bound_Y = tuple([tuple(bound_along_one_axis)
                              for bound_along_one_axis in np.array([self.lb_Y, self.ub_Y]).T.tolist()])
        self.len_Y = self.ub_Y - self.lb_Y
        self.constraint_space = gym.spaces.Box(low=np.float32(self.lb_Y),
                                               high=np.float32(self.ub_Y),
                                               shape=(self.dim_Y,),
                                               dtype=np.float32)

        # Init y0 for algorithm.
        # FIXME: The following code is only for the experiment.
        self.y0 = np.array([0.3, 0.7])

        self.grid = Grid(dim_Y=self.dim_Y, lb_Y=self.lb_Y, ub_Y=self.ub_Y)
        self.check_grid = Grid(dim_Y=self.dim_Y, lb_Y=self.lb_Y, ub_Y=self.ub_Y)

        # Joint space of observation and y.
        self.joint_obs_y_space: gym.Env = None
        self.max_episode_steps = env_config['max_steps']
        return

    # Sample a single y uniformly.
    def sample_y(self) -> ndarray:
        # Return shape (dim_Y,).
        return self.sample_batch_y(y_num=1)[0]

    # Sample a batch of y uniformly.
    def sample_batch_y(self, y_num: int) -> ndarray:
        # Return shape (y_num, dim_Y).
        return self.rng.random(size=(y_num, self.dim_Y)) * self.len_Y + self.lb_Y

    def contain_y(self, y: ArrayType) -> bool:
        # y.shape (dim_Y,)
        return self.contain_batch_y(batch_y=y)

    def contain_batch_y(self, batch_y: ArrayType) -> bool:
        # batch_y.shape (y_num, dim_Y).
        if isinstance(batch_y, ndarray):
            return np.all((batch_y >= self.lb_Y) & (batch_y <= self.ub_Y))
        elif isinstance(batch_y, Tensor):
            return torch.all((batch_y >= self.lb_Y) & (batch_y <= self.ub_Y)).item()

    def batch_c(self, batch_y: ArrayType, batch_obs: ArrayType, batch_action: ArrayType) -> ArrayType:
        # Return shape: (traj_len, num_y)
        raise NotImplementedError

    def batch_u(self, batch_y: ArrayType) -> ArrayType:
        # Input shape: (num_y, dim_Y)
        # Return shape: (num_y,)
        raise NotImplementedError

    def c_func_on_traj(self, batch_obs: ArrayType, batch_action: ArrayType, y: ArrayType) -> ArrayType:
        # Input shape:
        # y: (dim_Y,)
        # batch_obs: (batch_size, obs_shape)
        # batch_action: (batch_size, action_shape)
        # Return shape: (batch_size,)
        assert y.ndim == 1
        return self.batch_c(batch_y=y[None, :], batch_obs=batch_obs, batch_action=batch_action).squeeze()

    def c_grad_on_traj(self, batch_obs: ArrayType, batch_action: ArrayType, y: ArrayType) -> ArrayType:
        # Return shape: (batch_size, dim_Y)
        raise NotImplementedError

    def c_hess_on_traj(self, batch_obs: ArrayType, batch_action: ArrayType, y: ArrayType) -> ArrayType:
        # Return shape: (batch_size, dim_Y, dim_Y)
        raise NotImplementedError

    def u_func(self, y: ndarray) -> float:
        assert isinstance(y, ndarray)
        assert y.ndim == 1
        return float(self.batch_u(batch_y=y[None, :]))

    def u_grad(self, y: ndarray) -> ndarray:
        # Return shape: (dim_Y,)
        raise NotImplementedError

    def u_hess(self, y: ndarray) -> ndarray:
        # Return shape: (dim_Y, dim_Y)
        raise NotImplementedError

    def reset(
        self,
        *args,
        seed=None,
        options=None
    ):
        if args:
            warnings.warn(f'Unexpected args: {args}.')
        self.seed = seed
        self.rng = default_rng(self.seed)
        self.grid.reset()
        self.check_grid.reset()
        return

    def set_seed(self, seed: int = None):
        self.seed = seed
        self.rng = default_rng(seed)
        return
