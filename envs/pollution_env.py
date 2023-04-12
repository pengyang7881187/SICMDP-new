import copy
import math
import numpy as np
import gymnasium as gym
import torch

from torch import Tensor
from numpy import pi, ndarray
from functools import partial
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

from typing import Optional, Callable
from ray.rllib.env.env_context import EnvContext

from utils import ArrayType
from .sicmdp_env import SICMDPEnv
from .pollution_env_utils import (
    get_pollute_func_grad_and_hess,
    get_protect_func_grad_and_hess
)


# Macro for indices of the action.
ANGLE_IDX = 0
STEP_SIZE_IDX = 1

# Marco for indices of the observation.
POS_X_IDX = 0
POS_Y_IDX = 1
ANGLE_TO_GOAL_IDX = 2
DIST_TO_GOAL_IDX = 3
ANGLE_TO_PROTECT_START_IDX = 4
DIST_TO_PROTECT_START_IDX = 5

# Numerical error for spacing.Box
BOX_EPS = 1e-6


# TODO: You can use VecEnv to speedup simulation.
# For simplicity, we only consider dim_Y = 2.
class PollutionEnv(SICMDPEnv):
    # This is a list of supported rendering modes, copy from navigation2DEnv
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}
    def __init__(self, env_config: EnvContext):
        env_config = copy.deepcopy(env_config)
        env_config['dim_Y'] = 2

        super().__init__(env_config)

        self.slope = (self.ub_Y[1] - self.lb_Y[1]) / (self.ub_Y[0] - self.lb_Y[0])
        
        self.constant_speed_flag = env_config['constant_speed_flag']
        self.symmetric_flag = env_config['symmetric_flag']

        self.start_pos = env_config['start_pos']  # (dim_Y,).
        self.goal_pos = env_config['goal_pos']  # (dim_Y,).
        if self.symmetric_flag:
            assert np.allclose(self.start_pos, self.lb_Y)
            assert np.allclose(self.goal_pos, self.ub_Y)

        self.max_distant = np.linalg.norm(self.len_Y) + BOX_EPS

        self.protect_poss = env_config['protect_poss']  # (num_protect_poss, dim_Y).
        self.num_protect_poss = self.protect_poss.shape[0]

        # Action: Angle [-pi, pi] and step size [0, 1].
        # Angle <-> Counterclockwise: [0, pi] <-> [0, 180], [-pi, 0] <-> [180, 360].
        self.action_angle_low = -pi
        self.action_angle_high = pi
        if self.symmetric_flag:
            self.action_angle_low = 0.
            self.action_angle_high = pi / 2
        self.action_step_low = 0.
        self.action_step_high = 1.
        if not self.constant_speed_flag:
            self.action_space = gym.spaces.Box(low=np.array([self.action_angle_low, self.action_step_low], dtype=np.float32),
                                               high=np.array([self.action_angle_high, self.action_step_high], dtype=np.float32),
                                               shape=(2,),
                                               dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(low=np.array([self.action_angle_low,], dtype=np.float32),
                                               high=np.array([self.action_angle_high,], dtype=np.float32),
                                               shape=(1,),
                                               dtype=np.float32)

        # Observation:
        # (x_agent, y_agent, angle_to_goal, dis_to_goal, *[angle_to_protect_i, dis_to_protect_i]).
        self.lb_S = np.concatenate((self.lb_Y,
                                    [-pi, 0.] * (self.num_protect_poss + 1)), dtype=np.float32)
        self.ub_S = np.concatenate((self.ub_Y,
                                    [pi, self.max_distant] * (self.num_protect_poss + 1)), dtype=np.float32)
        self.dim_S = len(self.lb_S)

        self.observation_space = gym.spaces.Box(low=self.lb_S,
                                                high=self.ub_S,
                                                shape=(self.dim_S,),
                                                dtype=np.float32)

        self.joint_obs_y_space = gym.spaces.Dict({'obs': self.observation_space,
                                                  'y': self.constraint_space})

        # Actual movement: unit_speed * action[1].
        self.unit_speed = env_config['unit_speed']

        # Solve criterion: d(agent, goal) < goal_eps.
        self.goal_eps = env_config['goal_eps']

        # Cost of the action: step_cost_coeff * action[1] > 0.
        self.step_cost_coeff = env_config['step_cost_coeff']

        # Reward of reaching the goal.
        self.goal_reward = env_config['goal_reward']
        self.reward_range = (-self.step_cost_coeff * self.action_step_high, self.goal_reward)

        # Setting of c and u.
        self.pollute_pos_influence_func_name = env_config['pollution_func_name']
        self.pollute_pos_influence_func_parameter = env_config['pollution_func_parameter']
        self.pollute_pos_influence_func, self.pollute_pos_influence_grad_d, self.pollute_pos_influence_hess_d = \
            get_pollute_func_grad_and_hess(
                self.pollute_pos_influence_func_name,
                self.pollute_pos_influence_func_parameter
            )

        self.protect_pos_tolerance_func_name = env_config['protect_func_name']
        self.protect_pos_tolerance_func_parameter = env_config['protect_func_parameter']
        self.protect_pos_tolerance_func_multiplier = env_config['protect_func_multiplier']
        self.protect_pos_tolerance_func_intercept = env_config['protect_func_intercept']
        self.protect_pos_tolerance_func, self.protect_pos_tolerance_grad_d, self.protect_pos_tolerance_hess_d = \
            get_protect_func_grad_and_hess(
                self.protect_pos_tolerance_func_name,
                self.protect_pos_tolerance_func_parameter,
                self.protect_pos_tolerance_func_multiplier,
                self.protect_pos_tolerance_func_intercept
            )

        # Agent position.
        self.agent_pos = self.start_pos.copy()

        # Record the agent trajectory inside the environment.
        self.record_pos_flag = env_config['record_pos_flag']
        self.record_pos_lst = [self.agent_pos.copy(), ]

        # Info
        self.step_cnt = 0

        # TODO: Implement rendering.
        self.screen_height = 600
        self.screen_width = 600
        self.viewer = None                  # Viewer for render()
        self.agent_trans = None             # Transform-object of the moving agent
        self.track_way = None               # Polyline object to draw the tracked way
        return

    # Only used in step.
    @staticmethod
    def get_batch_angle(pos1: ndarray, batch_pos2: ndarray) -> ndarray:
        assert pos1.ndim == 1 and batch_pos2.ndim == 2
        batch_vec = batch_pos2 - pos1
        return np.arctan2(batch_vec[:, 1], batch_vec[:, 0])

    # Only used in step.
    @staticmethod
    def get_angle(pos1: ndarray, pos2: ndarray) -> float:
        assert pos1.ndim == 1 and pos2.ndim == 1
        vec = pos2 - pos1
        return np.arctan2(vec[1], vec[0])

    # Only used in step.
    @staticmethod
    def get_batch_distance(pos1: ndarray, batch_pos2: ndarray) -> ndarray:
        assert pos1.ndim == 1 and batch_pos2.ndim == 2
        return np.linalg.norm(pos1 - batch_pos2, axis=1)

    # Only used in step.
    @staticmethod
    def get_distance(pos1: ndarray, pos2: ndarray) -> float:
        assert pos1.ndim == 1 and pos2.ndim == 1
        return np.linalg.norm(pos1 - pos2)

    # Support compute on gpu.
    @staticmethod
    def get_two_diff_batch_distance(first_batch_pos1: ArrayType, second_batch_pos2: ArrayType) \
            -> ArrayType:
        # first_batch_pos1 shape (N, dim_Y), second_batch_pos2 shape (M, dim_Y)
        # Return shape (N, M)
        # Typically, (1) N = trajectory length, M = y number,
        # or (2) N = y number, M = protect position number.

        # Check.
        assert first_batch_pos1.ndim == 2 and second_batch_pos2.ndim == 2, \
            (first_batch_pos1.ndim, second_batch_pos2.ndim)

        first_batch_pos1_axis = first_batch_pos1[:, None, :]
        if isinstance(first_batch_pos1, ndarray):
            return np.linalg.norm(first_batch_pos1_axis - second_batch_pos2, axis=2).astype(np.float32)
        elif isinstance(first_batch_pos1, Tensor):
            return torch.linalg.norm(first_batch_pos1_axis - second_batch_pos2, dim=2)
        else:
            import jax.numpy as jnp
            return jnp.linalg.norm(first_batch_pos1_axis - second_batch_pos2, axis=2)

    # Only used in step.
    def get_obs_from_agent_pos(self, agent_pos: Optional[ndarray] = None) -> ndarray:
        obs = np.empty(self.observation_space.shape, dtype=np.float32)
        # Agent position.
        if agent_pos is None:
            agent_pos = self.agent_pos
        obs[:self.dim_Y] = agent_pos  # This line will copy RHS implicitly.
        # Goal relative position.
        obs[ANGLE_TO_GOAL_IDX] = self.get_angle(pos1=agent_pos, pos2=self.goal_pos)
        obs[DIST_TO_GOAL_IDX] = self.get_distance(pos1=agent_pos, pos2=self.goal_pos)
        # Protect relative position.
        obs[ANGLE_TO_PROTECT_START_IDX::2] = self.get_batch_angle(pos1=agent_pos, batch_pos2=self.protect_poss)
        obs[DIST_TO_PROTECT_START_IDX::2] = self.get_batch_distance(pos1=agent_pos, batch_pos2=self.protect_poss)

        assert self.observation_space.contains(obs), obs
        return obs

    # Only used in step.
    def get_new_agent_pos(self, action, agent_pos: Optional[ndarray] = None):
        assert self.action_space.contains(action), action
        # Agent position.
        if agent_pos is None:
            agent_pos = self.agent_pos

        # ANGLE_IDX = 0
        angle = action[ANGLE_IDX]
        if not self.constant_speed_flag:
            step_size = action[STEP_SIZE_IDX] * self.unit_speed
        else:
            step_size = self.unit_speed

        new_agent_pos = np.empty((self.dim_Y,))

        # Calculate new position.
        new_agent_pos[0] = agent_pos[0] + np.cos(angle) * step_size
        new_agent_pos[1] = agent_pos[1] + np.sin(angle) * step_size

        # If symmetric flag.
        if self.symmetric_flag:
            new_agent_pos = self.project_to_upper_triangle(new_agent_pos)

        # Borders.
        new_agent_pos = np.maximum(np.minimum(self.ub_Y, new_agent_pos), self.lb_Y)
        return new_agent_pos

    # Get discretize path in a ndarray.
    def discretize_path(self, path: Callable[[float | ndarray], float | ndarray],
                              path_deri: Callable[[float], float]) -> ndarray:
        # We only support start at lb_Y and goal at up_Y
        assert np.allclose(self.start_pos, self.lb_Y)
        assert np.allclose(self.goal_pos, self.ub_Y)
        # In addition, we only support path is an increasing function.
        # And path must pass start and goal.

        speed = self.unit_speed

        def next_pos_func(pos_x: float, pos_y: float, x: float) -> float:
            return (path(x) - pos_y) ** 2. + (x - pos_x) ** 2. - speed ** 2.

        def next_pos_deri(pos_x: float, pos_y: float, x: float) -> float:
            return 2. * ((path(x) - pos_y) * path_deri(x) + x - pos_x)

        pos_lst = []
        current_pos = self.start_pos
        pos_lst.append(current_pos)

        while True:
            if self.get_distance(current_pos, self.goal_pos) < speed:
                pos_lst.append(self.goal_pos)
                break
            current_next_pos_func = partial(next_pos_func, current_pos[0], current_pos[1])
            current_next_pos_deri = partial(next_pos_deri, current_pos[0], current_pos[1])

            x_at_crossing = fsolve(func=current_next_pos_func, x0=(self.goal_pos[0]+current_pos[0])/2., fprime=current_next_pos_deri)[0]
            assert x_at_crossing > current_pos[0]
            current_pos = np.array([x_at_crossing, path(x_at_crossing)])
            pos_lst.append(current_pos)

        return np.array(pos_lst)

    # Evaluate a discrete path.
    def evaluate_along_discrete_path(self, discrete_path: ndarray, fineness: int, gamma: float, name: str):
        print(f'Gamma: {gamma}')
        check_grid_y = self.check_grid.generate(fineness=fineness)
        check_grid_y_reshape = check_grid_y.reshape(fineness+1, fineness+1, 2)
        X = check_grid_y_reshape[:, :, 0]
        Y = check_grid_y_reshape[:, :, 1]

        # Evaluate value function for reference.
        total_step = discrete_path.shape[0]
        if gamma < 1.:
            value_func = - np.sum((gamma ** np.arange(total_step - 1)) * (1. + np.linalg.norm(discrete_path - self.goal_pos[None, :], axis=1)) * self.step_cost_coeff) \
                     + (gamma ** total_step) * self.goal_reward
        elif math.isclose(gamma, 1.):
            value_func = -np.sum(1. + np.linalg.norm(discrete_path - self.goal_pos[None, :], axis=1)) * self.step_cost_coeff + self.goal_reward
        else:
            raise RuntimeError(f'Invalid gamma {gamma}!')
        print(f'Value function: {value_func}')

        # Evaluate constraint value function for determining u function.
        # Result shape (total_step, grid_y_size)
        grid_c_on_discrete_path = self.batch_c(batch_y=torch.as_tensor(check_grid_y, device=torch.device('cuda')),
                                               batch_obs=torch.as_tensor(discrete_path, device=torch.device('cuda')),
                                               batch_action=np.empty((1,)))
        grid_u = self.batch_u(batch_y=check_grid_y).reshape(fineness+1, fineness+1)
        discount_tensor = gamma ** torch.arange(total_step, device=grid_c_on_discrete_path.device)
        constraint_value_function_on_grid = torch.sum(discount_tensor[-1, None] * grid_c_on_discrete_path,
                                                      dim=0).cpu().numpy()
        c_v_func_reshape = constraint_value_function_on_grid.reshape(fineness+1, fineness+1)
        diff = grid_u - c_v_func_reshape  # > 0 <-> not violat

        # Plot surface
        plt.figure()
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot_surface(X, Y, diff, rstride=1, cstride=1, cmap='rainbow', label='u-Vc')
        plt.title(name)
        m = plt.cm.ScalarMappable(cmap='rainbow')
        m.set_array(diff)
        plt.colorbar(m)
        plt.show()
        print(f'max u {np.max(grid_u)}')
        print(f'min u {np.min(grid_u)}')
        print(f'max Vc {np.max(c_v_func_reshape)}')
        print(f'min Vc {np.min(c_v_func_reshape)}')
        print(f'max violat {-np.min(diff)}')
        print(f'min violat {-np.max(diff)}')
        print(f'violat proportion {np.sum((diff < 0).astype(np.float32)) / diff.size}')
        return

    def batch_c(self, batch_y: ArrayType, batch_obs: ArrayType, batch_action: ArrayType) -> ArrayType:
        # Return shape: (traj_len, num_y)
        assert batch_y.ndim == 2 and batch_obs.ndim == 2, (batch_y.ndim, batch_obs.ndim)
        batch_pos = batch_obs[:, :2]
        batch_pos_y_distant = self.get_two_diff_batch_distance(first_batch_pos1=batch_pos,
                                                               second_batch_pos2=batch_y)
        return self.pollute_pos_influence_func(batch_pos_y_distant)

    def c_grad_on_traj(self, batch_obs: ArrayType, batch_action: ArrayType, y: ArrayType) -> ArrayType:
        # Return shape: (batch_size, dim_Y)
        batch_pos = batch_obs[:, :2]
        batch_pos_y_distant = self.get_two_diff_batch_distance(first_batch_pos1=batch_pos,
                                                               second_batch_pos2=y[None, :]).squeeze()
        return (self.pollute_pos_influence_grad_d(batch_pos_y_distant) / batch_pos_y_distant)[:, None] * (y - batch_pos)

    def c_hess_on_traj(self, batch_obs: ArrayType, batch_action: ArrayType, y: ArrayType) -> ArrayType:
        # Return shape: (batch_size, dim_Y, dim_Y)
        batch_pos = batch_obs[:, :2]
        batch_y_minus_pos = y - batch_pos  # Shape (batch_size, dim_Y)
        # Shape (batch_size)
        batch_pos_y_distant = self.get_two_diff_batch_distance(first_batch_pos1=batch_pos,
                                                               second_batch_pos2=y[None, :]).squeeze()
        # Shape (batch_size, dim_Y)
        batch_partial_d_ys_to_partial_y = batch_y_minus_pos / batch_pos_y_distant[:, None]
        if isinstance(batch_obs, Tensor):
            # Shape (batch_size, dim_Y, dim_Y)
            batch_partial_d_ys_to_partial_y_outer = torch.einsum('bij, bjk->bik', batch_partial_d_ys_to_partial_y[:, :, None],
                                                                 batch_partial_d_ys_to_partial_y[:, None, :])
            identity = torch.eye(n=self.dim_Y, device=y.device)
        elif isinstance(batch_obs, ndarray):
            # Shape (batch_size, dim_Y, dim_Y)
            batch_partial_d_ys_to_partial_y_outer = np.einsum('bij, bjk->bik', batch_partial_d_ys_to_partial_y[:, :, None],
                                                              batch_partial_d_ys_to_partial_y[:, None, :])
            identity = np.eye(N=self.dim_Y)
        else:
            raise RuntimeError(f'Invalid input type {type(batch_obs)}!')
        batch_partial2_d_ys_to_partial_y2 = (identity - batch_partial_d_ys_to_partial_y_outer) / batch_pos_y_distant[:, None, None]
        return self.pollute_pos_influence_hess_d(batch_pos_y_distant)[:, None, None] * batch_partial_d_ys_to_partial_y_outer + \
            self.pollute_pos_influence_grad_d(batch_pos_y_distant)[:, None, None] * batch_partial2_d_ys_to_partial_y2

    def find_protect_pos_with_minimum_u(self, y: ndarray) -> ndarray:
        return self.protect_poss[np.argmin(self.protect_pos_tolerance_func(np.linalg.norm(y - self.protect_poss, axis=1)))]

    def u_grad(self, y: ndarray) -> ndarray:
        # Return shape: (dim_Y,)
        min_u_protect_pos = self.find_protect_pos_with_minimum_u(y)
        yp = y - min_u_protect_pos
        d_yp = np.linalg.norm(yp)
        if math.isclose(d_yp, 0.):
            return np.zeros_like(y)
        partial_d_yp_to_partial_y = yp / d_yp
        return self.protect_pos_tolerance_grad_d(d_yp) * partial_d_yp_to_partial_y

    def u_hess(self, y: ndarray) -> ndarray:
        # Return shape: (dim_Y, dim_Y)
        min_u_protect_pos = self.find_protect_pos_with_minimum_u(y)
        yp = y - min_u_protect_pos
        d_yp = np.linalg.norm(yp)
        partial_d_yp_to_partial_y = yp / d_yp
        partial_d_yp_to_partial_y_outer = partial_d_yp_to_partial_y[:, None] @ partial_d_yp_to_partial_y[None, :]
        identity = np.eye(N=self.dim_Y)
        partial2_d_yp_to_partial_y2 = (identity - partial_d_yp_to_partial_y_outer) / d_yp
        return self.protect_pos_tolerance_hess_d(d_yp) * partial_d_yp_to_partial_y_outer + \
            self.protect_pos_tolerance_grad_d(d_yp) * partial2_d_yp_to_partial_y2

    def batch_u(self, batch_y: ArrayType) -> ArrayType:
        # Return shape: (num_y,)
        # shape (num_y, num_protect)
        assert batch_y.ndim == 2
        batch_y_protect_distant = self.get_two_diff_batch_distance(first_batch_pos1=batch_y,
                                                                   second_batch_pos2=self.protect_poss)
        if isinstance(batch_y, ndarray):
            return np.min(self.protect_pos_tolerance_func(batch_y_protect_distant), axis=1)
        elif isinstance(batch_y, Tensor):
            result, _ = torch.min(self.protect_pos_tolerance_func(batch_y_protect_distant), dim=1)
            return result
        else:
            import jax.numpy as jnp
            return jnp.min(self.protect_pos_tolerance_func(batch_y_protect_distant), axis=1)

    def in_upper_triangle(self, agent_pos: Optional[ndarray] = None) -> bool:
        # Agent position.
        if agent_pos is None:
            agent_pos = self.agent_pos
        return agent_pos[1] >= self.slope * agent_pos[0]

    def project_to_upper_triangle(self, agent_pos: Optional[ndarray] = None) -> ndarray:
        # Agent position.
        if agent_pos is None:
            agent_pos = self.agent_pos
        if self.in_upper_triangle(agent_pos):
            return agent_pos
        else:
            e = np.array([1., self.slope])
            e = e / np.linalg.norm(e)
            project_pos = np.inner(e, agent_pos) * e
            return project_pos

    def step(self, action):
        self.step_cnt += 1
        self.agent_pos = self.get_new_agent_pos(action)
        if self.record_pos_flag:
            self.record_pos_lst.append(self.agent_pos.copy())
        obs = self.get_obs_from_agent_pos()
        terminated = False
        truncated = False
        if obs[DIST_TO_GOAL_IDX] <= self.goal_eps:
            # Solved.
            if self.record_pos_flag:
                self.record_pos_lst.append(self.goal_pos.copy())
            reward = self.goal_reward
            terminated = True
        else:
            reward = -self.step_cost_coeff * (obs[DIST_TO_GOAL_IDX] + 1.)

        # Break if step_cnt >= max_episode_steps.
        if self.step_cnt >= self.max_episode_steps:
            truncated = True

        info = {}

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *args,
        seed=None,
        options=None,
    ):
        super().reset(*args, seed=seed, options=options)
        self.agent_pos = self.start_pos.copy()
        self.record_pos_lst = [self.agent_pos.copy(), ]
        self.step_cnt = 0

        obs = self.get_obs_from_agent_pos()
        info = {}
        return obs, info

    # TODO: Implement rendering.
    def render(self):
        assert self.record_pos_flag
        if len(self.record_pos_lst) < 10:
            return
        print(self.action_space)
        discrete_path = np.array(self.record_pos_lst)
        movement_lst = np.linalg.norm(discrete_path[1:] - discrete_path[:-1], axis=1)
        print(movement_lst)
        plt.figure()
        plt.plot(discrete_path[:, 0], discrete_path[:, 1], marker='o', mec='none', ms=4, lw=1, label='Policy')
        plt.legend()
        plt.show()
        return

