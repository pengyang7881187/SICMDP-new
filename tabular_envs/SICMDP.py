import os
import time
import math
import torch
import warnings
import numpy as np
import gurobipy as gp
import gymnasium as gym

from tqdm import tqdm
from torch import Tensor
from gurobipy import GRB
from torch.linalg import solve
from scipy.optimize import minimize
from typing import Callable, Tuple, List


from .config import *
from utils.utility import torch_softmax, log, get_datetime_str, torch_vectorize_choice, \
    batch_torch_derivative_log_softmax, np_project_to_l2_ball

z_epsilon = 1e-16


class OptimizeError(Exception):
    def __init__(self, message='Fail to optimize.'):
        super(OptimizeError, self).__init__(message)
        return


# We only consider Y is a rectangular in R^d
class SICMDPEnv(gym.Env):
    def __init__(self, name: str, S: int, A: int, gamma: float, P: Tensor, r: Tensor,
                 c: Callable[[Tensor], Tensor], u: Callable[[Tensor], Tensor],
                 c_grad: Callable[[Tensor], Tensor], u_grad: Callable[[Tensor], Tensor],
                 c_hess: Callable[[Tensor], Tensor], u_hess: Callable[[Tensor], Tensor],
                 mu: Tensor, dim_Y: int, lb_Y: Tensor, ub_Y: Tensor, device: torch.device, *args, **kwargs):
        super(SICMDPEnv, self).__init__(*args, **kwargs)
        self.name = name
        self.device = device
        self.S = S
        self.A = A
        self.gamma = gamma
        self.P = torch.as_tensor(P, dtype=torch.float32, device=self.device)  # (S, A, S)
        self.r = torch.as_tensor(r, dtype=torch.float32, device=self.device)  # (S, A)
        # Both c and u support vectorization, see toymdp_env for example
        self.c = c  # Given y, c(y) is a Tensor with shape (S, A) or (y_num, S, A)
        self.u = u  # Given y, u(y) is a real number or (y_num)
        self.c_grad = c_grad
        self.u_grad = u_grad
        self.c_hess = c_hess
        self.u_hess = u_hess
        self.mu = torch.as_tensor(mu, dtype=torch.float32, device=self.device)  # Initial distribution
        self.dim_Y = dim_Y  # Dimension of Y
        # Range of rectangular Y
        self.lb_Y = torch.as_tensor(lb_Y, dtype=torch.float32, device=self.device)
        self.ub_Y = torch.as_tensor(ub_Y, dtype=torch.float32, device=self.device)
        # Used in bound argument in scipy.optimize.minimize.
        self.bound_Y = tuple([tuple(bound_along_one_axis)
                              for bound_along_one_axis in np.array([self.lb_Y.cpu().numpy(), self.ub_Y.cpu().numpy()]).T.tolist()])
        # Initial y0 for algorithm
        self.y0 = (self.lb_Y + self.ub_Y) / 2.
        self.init_y_for_optimize = self.y0.cpu().numpy().copy()
        # self.y0 = self.lb_Y
        # P, R, mu, lb_Y and ub_Y should be valid
        assert self.P.shape == (self.S, self.A, self.S)
        assert torch.allclose(torch.sum(self.P, dim=2), torch.ones((self.S, self.A), device=self.device))
        assert self.mu.shape == (self.S,)
        assert np.allclose(torch.sum(self.mu).cpu(), 1.)
        assert self.r.shape == (self.S, self.A)
        assert self.lb_Y.shape == (self.dim_Y,)
        assert self.ub_Y.shape == (self.dim_Y,)

        self.empty_cache_flag = False

        self.vals_mask = torch.tensor([0], dtype=torch.int, device=self.device)

        # num_grid_nodes = (fineness + 1) ^ dim_Y
        self.grid_fineness = -1  # -1 is a negative flag,
        self.grid_c_u_flag = False  # Flag indicates whether c and u on the grid have been computed
        self.check_grid_fineness = -1  # -1 is a negative flag
        self.check_grid_c_u_flag = False  # Same as grid_c_u_flag

        self.grid = torch.zeros((1), device=self.device)
        self.check_grid = torch.zeros((1), device=self.device)
        self.grid_c = torch.zeros((1), device=self.device)
        self.grid_u = torch.zeros((1), device=self.device)
        self.check_grid_c = torch.zeros((1), device=self.device)
        self.check_grid_u = torch.zeros((1), device=self.device)

    # Reset grid for training
    def reset_grid(self):
        self.grid_fineness = -1  # -1 is a negative flag
        self.grid_c_u_flag = False
        self.grid = torch.zeros((1), device=self.device)
        self.grid_c = torch.zeros((1), device=self.device)
        self.grid_u = torch.zeros((1), device=self.device)
        return

    # Reset grid for validation
    def reset_check_grid(self):
        self.check_grid_fineness = -1  # -1 is a negative flag
        self.check_grid_c_u_flag = False
        self.check_grid = torch.zeros((1), device=self.device)
        self.check_grid_c = torch.zeros((1), device=self.device)
        self.check_grid_u = torch.zeros((1), device=self.device)
        return

    # Check whether an input policy pi is a valid policy
    def check_pi(self, pi: Tensor):
        assert pi.shape == (self.S, self.A)
        assert torch.allclose(torch.sum(pi, dim=1), torch.ones((self.S), device=self.device))
        return

    # Old code from NIPS22
    # Check whether an input z is feasible and find new y via discretization (P is not known)
    def check_z_feasible_and_find_new_y(self, z: Tensor, grid: Tensor, grid_c: Tensor, grid_u: Tensor,
                                        epsilon=1e-12) -> Tuple[bool, Tensor, float]:
        feasible_flag = False
        reduce_z = torch.sum(torch.as_tensor(z, dtype=torch.float32, device=self.device), dim=2)
        vals = (1. / (1. - self.gamma)) * torch.sum(grid_c * reduce_z, dim=(1, 2)) - grid_u
        vals = vals + self.vals_mask
        max_y_index = torch.argmax(vals)
        max_cons_violat = vals[max_y_index]
        max_y = grid[max_y_index]
        self.vals_mask[max_y_index] = -np.inf
        if max_cons_violat <= epsilon:
            feasible_flag = True
        return feasible_flag, max_y, float(max_cons_violat)

    # Old code from NIPS22
    # Check whether a policy pi is feasible (P is known)
    def check_pi_feasible_true_P(self, pi: Tensor, check_fineness: int, epsilon=1e-10) -> Tuple[bool, float]:
        grid = self.generate_grid(fineness=check_fineness, check_flag=True)
        grid_c, grid_u = self.generate_grid_c_u(check_flag=True)
        feasible_flag = False
        q_pi = self.q_pi(pi)
        vals = (1. / (1. - self.gamma)) * torch.sum(grid_c * q_pi, dim=(1, 2)) - grid_u
        max_cons_violat = torch.max(vals)
        # Small tolerance (Avoid numerical error)
        if max_cons_violat <= epsilon:
            feasible_flag = True
        return feasible_flag, float(max_cons_violat)

    # heat map for violation of constraints.
    def generate_heat_map(self, pi: Tensor, check_fineness: int) -> Tensor:
        grid = self.generate_grid(fineness=check_fineness, check_flag=True)
        grid_c, grid_u = self.generate_grid_c_u(check_flag=True)
        q_pi = self.q_pi(pi)
        vals = torch.maximum((1. / (1. - self.gamma)) * torch.sum(grid_c * q_pi, dim=(1, 2)) - grid_u,
                             torch.zeros((grid_c.shape[0]), device=self.device))
        return vals

    # Generate grids of Y
    def generate_grid(self, fineness=100, check_flag=False) -> Tensor:
        if check_flag:
            # The check grid has been computed
            if self.check_grid_fineness == fineness:
                return self.check_grid
            elif self.check_grid_fineness != -1:
                self.reset_check_grid()
        else:
            # Initialize the array which avoid selecting the same grid as before
            self.vals_mask = torch.zeros(int((fineness+1) ** self.dim_Y), device=self.device)
            # The grid has been computed
            if self.grid_fineness == fineness:
                return self.grid
            elif self.grid_fineness != -1:
                self.reset_grid()

        # Generate coordinate set for the grid
        grid_num = fineness + 1
        coordinate_lst = []
        index_array = torch.tensor(np.indices(list(self.dim_Y * [grid_num])), device=self.device)
        for i in range(self.dim_Y):
            coordinate_lst.append(index_array[i].reshape(-1) * ((self.ub_Y[i] - self.lb_Y[i]) / fineness) + self.lb_Y[i])
        grid = torch.stack(list(coordinate_lst), dim=1).to(self.device)

        # Save the result
        if check_flag:
            self.check_grid = grid
            self.check_grid_fineness = fineness
        else:
            self.grid = grid
            self.grid_fineness = fineness
        return grid

    # When you call this method, the grid must have been generated
    def generate_grid_c_u(self, check_flag=False) -> Tuple[Tensor, Tensor]:
        if check_flag:
            if self.check_grid_c_u_flag:
                return self.check_grid_c, self.check_grid_u
            grid = self.check_grid
        else:
            if self.grid_c_u_flag:
                return self.grid_c, self.grid_u
            grid = self.grid

        grid_c = self.c(grid)
        grid_u = self.u(grid)

        # Save the result
        if check_flag:
            self.check_grid_c = grid_c
            self.check_grid_u = grid_u
            self.check_grid_c_u_flag = True
        else:
            self.grid_c = grid_c
            self.grid_u = grid_u
            self.grid_c_u_flag = True
        return grid_c, grid_u

    # (S,)
    def r_pi(self, pi: Tensor) -> Tensor:
        self.check_pi(pi)
        return torch.sum(pi * self.r, dim=1)

    # (S,) or (y_num, S)
    def c_pi_y(self, pi: Tensor, y: Tensor) -> Tensor:
        self.check_pi(pi)
        return torch.sum(pi * self.c(y), dim=-1)

    # (S, S)
    def P_pi(self, pi: Tensor) -> Tensor:
        self.check_pi(pi)
        pi_axis = pi[:, :, None]
        return torch.sum(self.P * pi_axis, dim=1)

    # Value function, (S,)
    def V_pi(self, pi: Tensor) -> Tensor:
        r_pi = self.r_pi(pi)
        P_pi = self.P_pi(pi)
        return solve(torch.eye(self.S, device=self.device) - self.gamma * P_pi, r_pi)

    # Obj_pi = V_pi(mu)
    def Obj_pi(self, pi: Tensor) -> float:
        V_pi = self.V_pi(pi)
        return float(torch.inner(V_pi, self.mu))

    # Constraint value function, (S,) or (y_num, S)
    def C_pi_y(self, pi: Tensor, y: Tensor) -> Tensor:
        c_pi_y = self.c_pi_y(pi, y).T  # (S,) or (S, y_num)
        P_pi = self.P_pi(pi)  # (S, S)
        return solve(torch.eye(self.S, device=self.device) - self.gamma * P_pi, c_pi_y).T

    # float or (y_num,)
    def V_pi_cy(self, pi: Tensor, y: Tensor) -> Tensor | float:
        C_pi_y = self.C_pi_y(pi, y)
        return torch.inner(C_pi_y, self.mu)

    # Q value function, (S, A)
    def Q_pi(self, pi: Tensor) -> Tensor:
        V_pi = self.V_pi(pi)
        return self.r + self.gamma * torch.inner(self.P, V_pi)

    # Advantage of function b of shape (S, A), s_array and a_array of shape (batch_size,)
    # Return shape (batch_size,)
    def A_pi_b(self, pi: Tensor, b: Tensor, s_array: Tensor, a_array: Tensor) -> Tensor:
        b_pi = torch.sum(pi * b, dim=1)  # (S,)
        P_pi = self.P_pi(pi)  # (S, S)
        V_pi_b = solve(torch.eye(self.S, device=self.device) - self.gamma * P_pi, b_pi)  # (S,)
        Q_pi_b = b + self.gamma * torch.inner(self.P, V_pi_b)  # (S, A)
        Q_b = Q_pi_b[s_array, a_array]  # (batch_size,)
        V_b = V_pi_b[s_array]  # (batch_size,)
        return Q_b - V_b

    # State occupancy measure, (S,)
    def d_pi(self, pi: Tensor) -> Tensor:
        P_pi = self.P_pi(pi)
        return solve((torch.eye(self.S, device=self.device) - self.gamma * P_pi).T, self.mu) * (1 - self.gamma)

    # Old code from NIPS22
    # State-action occupancy measure, (S, A)
    def q_pi(self, pi: Tensor) -> Tensor:
        d_pi = self.d_pi(pi)
        return pi * d_pi[:, None]

    # Old code from NIPS22
    # State-action-state occupancy measure, (S, A, S)
    def z_pi(self, pi: Tensor) -> Tensor:
        q_pi = self.q_pi(pi)
        return self.P * q_pi[:, :, None]

    # Old code from NIPS22
    # Recover policy pi from z_pi
    def pi_z(self, z: Tensor) -> Tensor:
        q = torch.sum(z, dim=2)
        return q / torch.sum(q, dim=1)[:, None]

    # Old code from NIPS22
    # Recover P_UC from z
    def P_z(self, z: Tensor) -> Tensor:
        q = torch.sum(z, dim=2)
        return z / q[:, :, None]

    # Old code from NIPS22
    # Sample from P, used in sample_uniformly and sample_from_nu
    # Given (s_t, a_t), sample s_(t+1)
    def sample_next_s(self, SA_array: Tensor) -> Tensor:
        SAS_array = torch.zeros((self.S, self.A, self.S), device=self.device)
        for s in range(self.S):
            for a in range(self.A):
                SAS_array[s, a, :] = torch.tensor(np.random.multinomial(n=SA_array[s, a], pvals=self.P[s, a]), device=self.device)
        return SAS_array

    # Old code from NIPS22
    # Sample uniformly in (S, A), m = n * S * A, then sample from P
    def sample_uniformly_state_action(self, n: int) -> Tuple[Tensor, Tensor]:
        SA_array = n * torch.ones((self.S, self.A), dtype=torch.long, device=self.device)
        return SA_array, self.sample_next_s(SA_array)

    # Old code from NIPS22
    # Sample m times from nu, which is a distribution on (S, A), then sample from P
    def sample_from_nu_state_action(self, m: int, nu: Tensor) -> Tuple[Tensor, Tensor]:
        assert nu.shape == (self.S, self.A)
        assert np.allclose(torch.sum(nu), 1.)
        nu = nu.reshape(-1)
        SA_array = torch.tensor(np.random.multinomial(n=m, pvals=nu).reshape(self.S, self.A), device=self.device)
        return SA_array, self.sample_next_s(SA_array)

    # Old code from NIPS22
    # Check whether P is in the uncertainty set
    def check_uncertainty_set(self, P_hat: Tensor, d_delta: float) -> Tensor:
        nan = torch.isnan(d_delta).to(self.device)
        ub = (P_hat + d_delta > self.P)
        lb = (P_hat - d_delta < self.P)
        return torch.all((ub & lb) | nan)

    # Old code from NIPS22
    # 用高精度进行离散化，将SICMDP转换为CMDP，用线性规划的方式求解
    def SI_plan(self, iter_upper_bound: int, fineness: int, check_fineness: int, silent_flag=True)\
            -> Tuple[bool, Tensor, Tensor, List[Tensor], float, float]:
        search_grid = self.generate_grid(fineness=fineness)
        search_grid_c, search_grid_u = self.generate_grid_c_u(check_flag=False)

        time_start = time.time()
        S = self.S
        A = self.A

        Y0 = [self.y0]
        z = np.zeros((S, A, S))
        old_z = np.zeros((S, A, S))

        # Model with gurobi
        model = gp.Model(self.name + ' Extended LSIP')

        model.setParam('OutputFlag', 0)

        z_var = model.addVars(range(S), range(A), range(S), lb=0, name='z')
        # Set objective
        obj = gp.LinExpr()
        for s in range(S):
            for a in range(A):
                obj += (z_var.sum(s, a, '*') * self.r[s, a])
        model.setObjective(obj, GRB.MAXIMIZE)

        ##### Set constraints #####
        # y0
        constr_y0 = gp.LinExpr()
        for s in range(S):
            for a in range(A):
                constr_y0 += (1. / (1. - self.gamma)) * (z_var.sum(s, a, '*') * self.c(self.y0)[s, a])
        model.addConstr(constr_y0 <= self.u(self.y0), name='c_y0')
        # MDP
        for s in range(S):
            for a in range(A):
                for s1 in range(S):
                    name = 'c_' + str(s) + '_' + str(a) + '_' + str(s1) + '_p'
                    model.addConstr(z_var[s, a, s1] - self.P[s, a, s1] * z_var.sum(s, a, '*') == 0, name=name)
        # Valid occupancy measure
        for s in range(S):
            name = 'c_valid_measure_' + str(s)
            model.addConstr(z_var.sum(s, '*', '*') - (1 - self.gamma) * self.mu[s] - self.gamma * z_var.sum('*', '*', s) == 0,
                            name=name)
        #########################
        Y0_size = 1
        for i in range(iter_upper_bound):
            if not silent_flag:
                print('Step ' + str(i + 1) + ':')
            # Solve LP, if infeasible: return, the problem is infeasible
            model.optimize()
            if model.status == GRB.INFEASIBLE or model.status == GRB.INF_OR_UNBD:
                raise RuntimeError
                # return False, -1, -1, -1, np.inf, -np.inf
            if model.status != GRB.OPTIMAL:
                print('Status: ' + str(model.status))
                raise RuntimeError
                # break
            for s in range(S):
                for a in range(A):
                    for s1 in range(S):
                        z[s, a, s1] = z_var[s, a, s1].X

            # Is z feasible for SICMDP? Feasible: break, z is optimal; Infeasible: continue
            # At the same time, add new y to Y0 and modify the model
            feasible_flag, y, max_cons_violat = self.check_z_feasible_and_find_new_y(torch.from_numpy(z),
                                                                                     search_grid, search_grid_c,
                                                                                     search_grid_u)
            Obj = model.ObjVal  # This is not the true Obj for the RL problem, but the LP OBJ
            if not silent_flag:
                print('Obj: %g' % Obj)
                print('Max constraint violation: %g' % max_cons_violat)
            # Policies could be the same, but the chosen Ps from the uncertainty set could be different
            # print('Policy:' + str(envs.pi_z(z)))

            if feasible_flag:
                break
            if np.linalg.norm(z - old_z) < z_epsilon:
                if not silent_flag:
                    print('z not change')
                break
            old_z = z.copy()
            Y0.append(y)
            Y0_size += 1
            constr_y = gp.LinExpr()
            for s in range(S):
                for a in range(A):
                    constr_y += (1. / (1. - self.gamma)) * (z_var.sum(s, a, '*') * self.c(y)[s, a])
            constr_name = 'c_y' + str(Y0_size)
            model.addConstr(constr_y <= self.u(y), name=constr_name)
        time_end = time.time()
        if not silent_flag:
            print('Time: ' + str(time_end - time_start))
            print('Checking...')

        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        pi_hat = self.pi_z(z)
        feasible_flag, max_cons_violat = self.check_pi_feasible_true_P(pi_hat, check_fineness)
        Obj = self.Obj_pi(pi_hat)
        return feasible_flag, pi_hat, z, Y0, float(max_cons_violat), float(Obj)

    ############################################################
    # New codes for SICPO

    # Find max violat y by optimization.
    def get_optimal_max_avg_monte_carlo_constraint_violat_value_and_pos(
            self, traj_s: Tensor, traj_a: Tensor
    ) -> Tuple[float, np.ndarray]:
        traj_num, traj_len = traj_s.shape
        gamma_tensor = self.gamma ** torch.arange(traj_len, device=self.device)
        class UMinusVcMonteCarlo:
            def __init__(inner_self):
                # TODO: Save intermediate results here to save time.
                return

            def func(inner_self, y: np.ndarray) -> float:
                if self.empty_cache_flag:
                    torch.cuda.empty_cache()
                y_tensor = torch.as_tensor(y, device=self.device)
                c_tensor = self.c(y_tensor)  # (S, A)
                c_traj_tensor = c_tensor[traj_s, traj_a]  # (traj_num, traj_len)
                total_c_traj_tensor = torch.sum(c_traj_tensor * gamma_tensor[None, :], dim=1)  # (traj_num)
                del c_traj_tensor
                if self.empty_cache_flag:
                    torch.cuda.empty_cache()
                V_c_hat_y = torch.mean(total_c_traj_tensor).item()
                del total_c_traj_tensor
                if self.empty_cache_flag:
                    torch.cuda.empty_cache()
                u_y = self.u(y_tensor).item()
                return u_y - V_c_hat_y

            def grad(inner_self, y: np.ndarray) -> np.ndarray:
                if self.empty_cache_flag:
                    torch.cuda.empty_cache()
                y_tensor = torch.as_tensor(y, device=self.device)
                c_grad_tensor = self.c_grad(y_tensor)  # (S, A, dim_Y)
                c_grad_traj_tensor = c_grad_tensor[traj_s, traj_a]  # (traj_num, traj_len, dim_Y)
                total_c_grad_traj_tensor = torch.sum(c_grad_traj_tensor * gamma_tensor[None, :, None], dim=1)  # (traj_num, dim_Y)
                del c_grad_traj_tensor
                if self.empty_cache_flag:
                    torch.cuda.empty_cache()
                V_c_hat_grad_y = torch.mean(total_c_grad_traj_tensor, dim=0).cpu().numpy()
                del total_c_grad_traj_tensor
                if self.empty_cache_flag:
                    torch.cuda.empty_cache()
                u_y_grad = self.u_grad(y_tensor).cpu().numpy()
                return u_y_grad - V_c_hat_grad_y

            def hess(inner_self, y: np.ndarray) -> np.ndarray:
                if self.empty_cache_flag:
                    torch.cuda.empty_cache()
                y_tensor = torch.as_tensor(y, device=self.device)
                c_hess_tensor = self.c_hess(y_tensor)  # (S, A, dim_Y)
                c_hess_traj_tensor = c_hess_tensor[traj_s, traj_a]  # (traj_num, traj_len, dim_Y, dim_Y)
                total_c_hess_traj_tensor = torch.sum(c_hess_traj_tensor * gamma_tensor[None, :, None, None], dim=1)  # (traj_num, dim_Y, dim_Y)
                del c_hess_traj_tensor
                if self.empty_cache_flag:
                    torch.cuda.empty_cache()
                V_c_hat_hess_y = torch.mean(total_c_hess_traj_tensor, dim=0).cpu().numpy()
                del total_c_hess_traj_tensor
                if self.empty_cache_flag:
                    torch.cuda.empty_cache()
                u_y_hess = self.u_hess(y_tensor).cpu().numpy()
                return u_y_hess - V_c_hat_hess_y

        u_minus_Vc_monte_carlo_instance = UMinusVcMonteCarlo()
        opt_res = minimize(
            fun=u_minus_Vc_monte_carlo_instance.func,
            x0=self.init_y_for_optimize,
            jac=u_minus_Vc_monte_carlo_instance.grad,
            hess=u_minus_Vc_monte_carlo_instance.hess,
            bounds=self.bound_Y,
            method='trust-constr',
        )
        if not opt_res.success:
            print('Optimization result:')
            print(opt_res)
            raise OptimizeError('Fail to find the max violat y.')
        max_avg_constraint_value_violat_y = opt_res.x.copy()
        self.init_y_for_optimize = opt_res.x.copy()
        max_avg_constraint_value_violat = -opt_res.fun

        return max_avg_constraint_value_violat, max_avg_constraint_value_violat_y

    # (y_num, dim_Y) Sample a list of y uniformly.
    def sample_y(self, y_num: int) -> Tensor:
        return torch.rand((y_num, self.dim_Y), device=self.device) * (self.ub_Y - self.lb_Y) + self.lb_Y

    # Sample trajectories.
    def sample_trajectories(self, pi: Tensor, traj_num: int, traj_len: int, init_flag=False,
                            init_s_array: Tensor = None, init_a_array: Tensor = None) -> Tuple[Tensor, Tensor]:
        self.check_pi(pi)
        # This shape is Memory-friendly
        traj_s = torch.empty((traj_len, traj_num), dtype=torch.long, device=self.device)
        traj_a = torch.empty((traj_len, traj_num), dtype=torch.long, device=self.device)
        if init_flag:
            # (s0, a0) is provided
            traj_s[0, :] = init_s_array
            traj_a[0, :] = init_a_array
        else:
            # Sample s0 from mu
            traj_s[0, :] = torch.tensor(np.random.choice(self.S, size=traj_num, replace=True, p=self.mu.cpu()), device=self.device)
            # Sample a0 from s0 and put it on device
            traj_a[0, :] = torch_vectorize_choice(p_tensor=pi[traj_s[0, :], :], device=self.device)
        # Simulate trajectories with vectorization
        for t in range(traj_len-1):
            # Sample s_{t+1} from s_t and a_t
            traj_s[t+1, :] = torch_vectorize_choice(p_tensor=self.P[traj_s[t, :], traj_a[t, :], :], device=self.device)
            # Sample a_{t+1} from s_{t+1} and pi
            traj_a[t+1, :] = torch_vectorize_choice(p_tensor=pi[traj_s[t+1, :], :], device=self.device)
        return traj_s.T, traj_a.T  # Return shape (traj_num, traj_len)

    # Return (y_num,), V^pi_cy(mu) based on trajectories
    def sample_based_V_pi_cy_set_mu(self, y_set: Tensor, traj_s: Tensor, traj_a: Tensor) -> Tensor:
        c_array = self.c(y_set)  # (y_num, S, A)
        traj_num, traj_len = traj_s.shape
        gamma_array = self.gamma ** torch.arange(traj_len, device=self.device)  # (traj_len,)
        c_traj_array = c_array[:, traj_s, traj_a]  # (y_num, traj_num, traj_len)
        total_c_traj_array = torch.sum(c_traj_array * gamma_array, dim=2)  # (y_num, traj_num)
        V_c_hat_array = torch.mean(total_c_traj_array, dim=1)  # (y_num,)
        return V_c_hat_array

    # Precise implementation of sample-based NPG
    # Only work for softmax parametrization
    # b.shape == (S, A)
    def sample_based_NPG(self, pi: Tensor, train_Q_traj_s: Tensor, train_Q_traj_a: Tensor, train_V_traj_s: Tensor,
                         train_V_traj_a: Tensor, b: Tensor, W: float, inner_init_lr: float,
                         gt_advantage_flag=False) -> Tensor:
        # You should make sure softmax(pi_logit) == pi
        train_traj_num, train_traj_len = train_Q_traj_s.shape
        train_init_s_array = train_Q_traj_s[:, 0]  # (train_traj_num,)
        train_init_a_array = train_Q_traj_a[:, 0]  # (train_traj_num,)
        self.check_pi(pi)
        if gt_advantage_flag:
            # Ground-truth advantage
            A_b_hat = self.A_pi_b(pi=pi, b=b, s_array=train_init_s_array, a_array=train_init_a_array)
        else:
            # Sample-based advantage
            gamma_array = self.gamma ** torch.arange(train_traj_len, device=self.device)  # (train_traj_len,)
            Q_b_hat = torch.sum(b[train_Q_traj_s, train_Q_traj_a] * gamma_array, dim=1)  # (train_traj_num,)
            V_b_hat = torch.sum(b[train_V_traj_s, train_V_traj_a] * gamma_array, dim=1)  # (train_traj_num,)
            A_b_hat = Q_b_hat - V_b_hat

        score_pi = batch_torch_derivative_log_softmax(prob=pi, s_array=train_init_s_array, a_array=train_init_a_array,
                                                      device=self.device)

        lr = inner_init_lr  # eta_0

        # numpy is faster here.
        A_b_hat = A_b_hat.cpu().numpy()
        score_pi = score_pi.cpu().numpy()
        w_array = np.zeros((train_traj_num, self.S, self.A))
        G0 = -2. * (1 - self.gamma) * A_b_hat[0] * score_pi[0]
        w_array[0] = np_project_to_l2_ball(-lr * G0, W)
        for k in range(train_traj_num - 1):
            lr *= ((k + 1) / (k + 2))  # eta_{k+1}
            G = 2. * (1 - self.gamma) * (np.tensordot(score_pi[k+1], w_array[k]) * (1 - self.gamma) -
                                         A_b_hat[k+1]) * score_pi[k+1]
            w_array[k+1] = np_project_to_l2_ball(w_array[k] - lr * G, W)
        w_array = torch.as_tensor(w_array, device=self.device)
        gamma_k_array = ((2. * torch.arange(train_traj_num, device=self.device) + 1.) / (train_traj_num * (train_traj_num + 1))).reshape((-1, 1, 1))
        return torch.sum(gamma_k_array * w_array, dim=0)

        # torch implementation
        # w_array = torch.zeros((train_traj_num, self.S, self.A), device=self.device)
        # G0 = -2. * (1 - self.gamma) * A_b_hat[0] * score_pi[0]
        # w_array[0] = torch_project_to_l2_ball(-lr * G0, W)
        # for k in range(train_traj_num - 1):
        #     lr *= ((k + 1) / (k + 2))  # eta_{k+1}
        #     G = 2. * (1 - self.gamma) * (torch.tensordot(score_pi[k+1], w_array[k]) * (1 - self.gamma) -
        #                                  A_b_hat[k+1]) * score_pi[k+1]
        #     w_array[k+1] = torch_project_to_l2_ball(w_array[k] - lr * G, W)
        # gamma_k_array = ((2. * torch.arange(train_traj_num, device=self.device) + 1.) / (train_traj_num * (train_traj_num + 1))).reshape((-1, 1, 1))
        # return torch.sum(gamma_k_array * w_array, dim=0)

    # iter_upper_bound: T
    # y_size: M
    # lr: alpha
    # eta: threshold for switching between reward and constraint
    # traj_num: K
    # traj_len: H
    def SICPO(self, exp_name: str, log_dir: str, eta=ETA, lr=LR_COEFF / math.sqrt(ITER_UPPER_BOUND), silent_flag=False,
              iter_upper_bound=ITER_UPPER_BOUND, y_size=SAMPLE_Y_SIZE,
              traj_num=TRAJ_NUM, traj_len=TRAJ_LEN,
              train_traj_num=TRAIN_TRAJ_NUM, train_traj_len=TRAIN_TRAJ_LEN,
              inner_init_lr=INNER_INIT_LR, W=DEFAULT_W, pi_mode=PI_MODE,
              gt_evaluate_flag=False, gt_advantage_flag=False,
              log_evaluate_flag=False, check_fineness=CHECK_FINENESS,
              optimize_y_flag=False
              ):
        # Logging
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logfile = os.path.join(log_dir, f'log_{exp_name}.txt')
        log(f'Current time: {get_datetime_str()}', logfile, silent_flag)
        log(f'Exp name: {exp_name}', logfile, silent_flag)
        log(f'Env name: {self.name}', logfile, silent_flag)
        log(f'Optimize y flag: {optimize_y_flag}', logfile, silent_flag)
        log(f'Threshold eta: {eta}', logfile, silent_flag)
        log(f'Learning rate alpha: {lr}', logfile, silent_flag)
        log(f'Inner loop initial learning rate eta_0: {inner_init_lr}', logfile, silent_flag)
        log(f'Radius of the ball W: {W}', logfile, silent_flag)
        log(f'Sample y size M: {y_size}', logfile, silent_flag)
        log(f'Iter upper bound T: {iter_upper_bound}', logfile, silent_flag)
        log(f'Trajectory number K_eval: {traj_num}', logfile, silent_flag)
        log(f'Trajectory length H: {traj_len}', logfile, silent_flag)
        log(f'Train trajectory number K_sgd: {train_traj_num}', logfile, silent_flag)
        log(f'Train trajectory length H_sgd: {train_traj_len}', logfile, silent_flag)  # Same as H in the paper
        log(f'Return policy mode: {pi_mode}', logfile, silent_flag)
        log(f'Ground-truth evaluate flag: {gt_evaluate_flag}', logfile, silent_flag)  # Use gt to evaluate error
        log(f'Ground-truth advantage flag: {gt_advantage_flag}', logfile, silent_flag)  # Use gt to compute advantage
        log(f'Log evaluate flag: {log_evaluate_flag}', logfile, silent_flag)  # Record error at each iteration
        log(f'Check fineness: {check_fineness}', logfile, silent_flag)  # Check fineness for recording error

        # B in the paper
        valid_pi_list = []

        pi_logit = torch.zeros((self.S, self.A), device=self.device)  # Use softmax parametrization by default

        total_time = 0

        true_Obj_array = -np.ones((iter_upper_bound))
        true_max_violat_array = -np.ones((iter_upper_bound))

        self.empty_cache_flag = False
        with tqdm(total=iter_upper_bound, desc=exp_name, unit='iter') as pbar:
            for i in range(iter_upper_bound):
                while True:
                    try:
                        old_pi_logit = pi_logit.clone()
                        valid_pi_list_update_flag = False

                        if self.empty_cache_flag:
                            torch.cuda.empty_cache()
                        start_time = time.time()

                        # Sample y
                        y_set = self.sample_y(y_size)

                        # Generate pi from pi_logit
                        pi = torch_softmax(pi_logit)

                        # Sample trajectories for evaluating constraints and selecting y_star
                        if gt_evaluate_flag:
                            traj_num = 100
                        # shape (traj_num, traj_len)
                        traj_s, traj_a = self.sample_trajectories(pi=pi, traj_num=traj_num, traj_len=traj_len)

                        # Evaluate constraints and select y_star
                        optimize_fail_flag = False
                        if optimize_y_flag:
                            assert not gt_evaluate_flag
                            try:
                                max_violate, y_star = \
                                    self.get_optimal_max_avg_monte_carlo_constraint_violat_value_and_pos(
                                        traj_s, traj_a
                                    )
                                y_star = torch.as_tensor(y_star, device=self.device)
                            except OptimizeError as optimize_error:
                                warnings.warn(f'Warning: Fail to optimize y with error information: {optimize_error}')
                                optimize_fail_flag = True
                                self.init_y_for_optimize = self.y0.cpu().numpy().copy()

                        if (not optimize_y_flag) or optimize_fail_flag:
                            u_array = self.u(y_set)
                            if gt_evaluate_flag:
                                V_c_hat_array = self.V_pi_cy(pi=pi, y=y_set)
                            else:
                                # Slow!
                                V_c_hat_array = self.sample_based_V_pi_cy_set_mu(y_set=y_set, traj_s=traj_s, traj_a=traj_a)
                                # shape (y_set,)
                            y_star_idx = torch.argmax(V_c_hat_array - u_array)
                            y_star = y_set[y_star_idx]
                            max_violate = V_c_hat_array[y_star_idx] - u_array[y_star_idx]

                        # Sample trajectories for NPG
                        # Collect initial (s0, a0) from the trajectories we collected before
                        train_init_traj_num_idx_array = torch.tensor(np.random.choice(a=traj_num, size=train_traj_num, replace=True), device=self.device)
                        train_init_traj_len_idx_array = torch.tensor(np.random.choice(a=traj_len, size=train_traj_num, replace=True), device=self.device)
                        train_init_s_array = traj_s[train_init_traj_num_idx_array, train_init_traj_len_idx_array]  # (train_traj_num,)
                        train_init_a_array = traj_a[train_init_traj_num_idx_array, train_init_traj_len_idx_array]  # (train_traj_num,)
                        del traj_s, traj_a
                        if self.empty_cache_flag:
                            torch.cuda.empty_cache()
                        if gt_advantage_flag:
                            # Ground-truth advantage
                            train_Q_traj_s = train_init_s_array.reshape(-1, 1)
                            train_Q_traj_a = train_init_a_array.reshape(-1, 1)
                            train_V_traj_s = train_Q_traj_s
                            train_V_traj_a = train_Q_traj_a  # Not used
                        else:
                            # Sample-based advantage
                            # Sample Q trajectories
                            train_Q_traj_s, train_Q_traj_a = self.sample_trajectories(pi=pi, traj_num=train_traj_num,
                                                                                      traj_len=train_traj_len, init_flag=True,
                                                                                      init_s_array=train_init_s_array,
                                                                                      init_a_array=train_init_a_array)
                            # shape (train_traj_num, train_traj_len)
                            # Sample V trajectories
                            # sample a0 from s0 and pi
                            V_train_init_a_array = torch_vectorize_choice(p_tensor=pi[train_init_s_array, :], device=self.device)  # (train_traj_num,)
                            train_V_traj_s, train_V_traj_a = self.sample_trajectories(pi=pi, traj_num=train_traj_num,
                                                                                      traj_len=train_traj_len, init_flag=True,
                                                                                      init_s_array=train_init_s_array,
                                                                                      init_a_array=V_train_init_a_array)
                            # shape (train_traj_num, train_traj_len)

                        # Update parameter
                        if max_violate <= eta:
                            # Update using reward
                            pi_logit += lr * self.sample_based_NPG(pi=pi, train_Q_traj_s=train_Q_traj_s, train_Q_traj_a=train_Q_traj_a,
                                                                   train_V_traj_s=train_V_traj_s, train_V_traj_a=train_V_traj_a,
                                                                   W=W, inner_init_lr=inner_init_lr,
                                                                   b=self.r, gt_advantage_flag=gt_advantage_flag)
                            # Add pi to B
                            valid_pi_list.append(pi)
                            valid_pi_list_update_flag = True
                        else:
                            # Update using y_star constraint
                            pi_logit -= lr * self.sample_based_NPG(pi=pi, train_Q_traj_s=train_Q_traj_s, train_Q_traj_a=train_Q_traj_a,
                                                                   train_V_traj_s=train_V_traj_s, train_V_traj_a=train_V_traj_a,
                                                                   W=W, inner_init_lr=inner_init_lr,
                                                                   b=self.c(y_star), gt_advantage_flag=gt_advantage_flag)

                        end_time = time.time()
                        total_time += (end_time - start_time)

                        del train_Q_traj_s, train_Q_traj_a, train_V_traj_s, train_V_traj_a
                        if self.empty_cache_flag:
                            torch.cuda.empty_cache()

                        # pi is not update yet here.
                        if log_evaluate_flag:
                            true_Obj_array[i] = self.Obj_pi(pi)
                            _, true_max_violat_array[i] = self.check_pi_feasible_true_P(pi, check_fineness)

                    except torch.cuda.OutOfMemoryError as oom_error:
                        warnings.warn(f'Warning: catch OOM error: {oom_error}')
                        if self.empty_cache_flag:
                            raise RuntimeError('Out of memory even if releasing the memory!')
                        self.empty_cache_flag = True
                        pi_logit = old_pi_logit
                        if valid_pi_list_update_flag:
                            valid_pi_list.pop()
                        continue

                    log(f'Iter {i}/{iter_upper_bound}: True Obj: {true_Obj_array[i]}, True Max Violate: {true_max_violat_array[i]}, '
                        f'Sample Max Violate: {max_violate} '
                        f'Time: {end_time - start_time}s, Total Time: {total_time}s, Valid size: {len(valid_pi_list)}',
                        logfile, silent_flag)
                    self.empty_cache_flag = False
                    pbar.update(1)
                    break
        if pi_mode == 'mean':
            final_pi = torch.mean(torch.tensor(valid_pi_list, device=self.device), dim=0)
        elif pi_mode == 'last':
            final_pi = valid_pi_list[-1]
        else:
            final_pi = torch.mean(torch.stack(valid_pi_list[-int(pi_mode):]), dim=0)
        return final_pi, true_Obj_array, true_max_violat_array

    def CRPO(self, exp_name: str, log_dir: str, eta=ETA, lr=LR_COEFF / math.sqrt(ITER_UPPER_BOUND), silent_flag=False,
             iter_upper_bound=ITER_UPPER_BOUND, grid_fineness=100,
             traj_num=TRAJ_NUM, traj_len=TRAJ_LEN,
             train_traj_num=TRAIN_TRAJ_NUM, train_traj_len=TRAIN_TRAJ_LEN,
             inner_init_lr=INNER_INIT_LR, W=DEFAULT_W, pi_mode=PI_MODE,
             gt_evaluate_flag=False, gt_advantage_flag=False,
             log_evaluate_flag=False, check_fineness=CHECK_FINENESS):
        # Logging
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logfile = os.path.join(log_dir, f'log_{exp_name}.txt')
        log(f'Current time: {get_datetime_str()}', logfile, silent_flag)
        log(f'Exp name: {exp_name}', logfile, silent_flag)
        log(f'Env name: {self.name}', logfile, silent_flag)
        log(f'Threshold eta: {eta}', logfile, silent_flag)
        log(f'Learning rate alpha: {lr}', logfile, silent_flag)
        log(f'Inner loop initial learning rate eta_0: {inner_init_lr}', logfile, silent_flag)
        log(f'Radius of the ball W: {W}', logfile, silent_flag)
        log(f'Grid fineness: {grid_fineness}', logfile, silent_flag)
        log(f'Iter upper bound T: {iter_upper_bound}', logfile, silent_flag)
        log(f'Trajectory number K_eval: {traj_num}', logfile, silent_flag)
        log(f'Trajectory length H: {traj_len}', logfile, silent_flag)
        log(f'Train trajectory number K_sgd: {train_traj_num}', logfile, silent_flag)
        log(f'Train trajectory length H_sgd: {train_traj_len}', logfile, silent_flag)  # Same as H in the paper
        log(f'Return policy mode: {pi_mode}', logfile, silent_flag)
        log(f'Ground-truth evaluate flag: {gt_evaluate_flag}', logfile, silent_flag)  # Use gt to evaluate error
        log(f'Ground-truth advantage flag: {gt_advantage_flag}', logfile, silent_flag)  # Use gt to compute advantage
        log(f'Log evaluate flag: {log_evaluate_flag}', logfile, silent_flag)  # Record error at each iteration
        log(f'Check fineness: {check_fineness}', logfile, silent_flag)  # Check fineness for recording error

        # B in the paper
        valid_pi_list = []

        pi_logit = torch.zeros((self.S, self.A), device=self.device)  # Use softmax parametrization by default

        total_time = 0

        true_Obj_array = -np.ones((iter_upper_bound))
        true_max_violat_array = -np.ones((iter_upper_bound))

        # Generate grid.
        y_set = self.generate_grid(grid_fineness)

        self.empty_cache_flag = False
        with tqdm(total=iter_upper_bound, desc=exp_name, unit='iter') as pbar:
            for i in range(iter_upper_bound):
                while True:
                    try:
                        old_pi_logit = pi_logit.clone()
                        valid_pi_list_update_flag = False

                        if self.empty_cache_flag:
                            torch.cuda.empty_cache()
                        start_time = time.time()

                        # Generate pi from pi_logit
                        pi = torch_softmax(pi_logit)

                        # Sample trajectories for evaluating constraints and selecting y_star
                        if gt_evaluate_flag:
                            traj_num = 100
                        traj_s, traj_a = self.sample_trajectories(pi=pi, traj_num=traj_num, traj_len=traj_len)
                        # shape (traj_num, traj_len)

                        # Evaluate constraints and select y_star
                        u_array = self.u(y_set)
                        if gt_evaluate_flag:
                            V_c_hat_array = self.V_pi_cy(pi=pi, y=y_set)
                        else:
                            # Slow!
                            V_c_hat_array = self.sample_based_V_pi_cy_set_mu(y_set=y_set, traj_s=traj_s, traj_a=traj_a)
                        # shape (y_set,)
                        y_star_idx = torch.argmax(V_c_hat_array - u_array)

                        # Sample trajectories for NPG
                        # Collect initial (s0, a0) from the trajectories we collected before
                        train_init_traj_num_idx_array = torch.tensor(np.random.choice(a=traj_num, size=train_traj_num, replace=True), device=self.device)
                        train_init_traj_len_idx_array = torch.tensor(np.random.choice(a=traj_len, size=train_traj_num, replace=True), device=self.device)
                        train_init_s_array = traj_s[train_init_traj_num_idx_array, train_init_traj_len_idx_array]  # (train_traj_num,)
                        train_init_a_array = traj_a[train_init_traj_num_idx_array, train_init_traj_len_idx_array]  # (train_traj_num,)
                        del traj_s, traj_a
                        if self.empty_cache_flag:
                            torch.cuda.empty_cache()
                        if gt_advantage_flag:
                            # Ground-truth advantage
                            train_Q_traj_s = train_init_s_array.reshape(-1, 1)
                            train_Q_traj_a = train_init_a_array.reshape(-1, 1)
                            train_V_traj_s = train_Q_traj_s
                            train_V_traj_a = train_Q_traj_a  # Not used
                        else:
                            # Sample-based advantage
                            # Sample Q trajectories
                            train_Q_traj_s, train_Q_traj_a = self.sample_trajectories(pi=pi, traj_num=train_traj_num,
                                                                                      traj_len=train_traj_len, init_flag=True,
                                                                                      init_s_array=train_init_s_array,
                                                                                      init_a_array=train_init_a_array)
                            # shape (train_traj_num, train_traj_len)
                            # Sample V trajectories
                            # sample a0 from s0 and pi
                            V_train_init_a_array = torch_vectorize_choice(p_tensor=pi[train_init_s_array, :], device=self.device)  # (train_traj_num,)
                            train_V_traj_s, train_V_traj_a = self.sample_trajectories(pi=pi, traj_num=train_traj_num,
                                                                                      traj_len=train_traj_len, init_flag=True,
                                                                                      init_s_array=train_init_s_array,
                                                                                      init_a_array=V_train_init_a_array)
                        # shape (train_traj_num, train_traj_len)

                        max_violate = V_c_hat_array[y_star_idx] - u_array[y_star_idx]

                        # Update parameter
                        if max_violate <= eta:
                            # Update using reward
                            pi_logit += lr * self.sample_based_NPG(pi=pi, train_Q_traj_s=train_Q_traj_s, train_Q_traj_a=train_Q_traj_a,
                                                                   train_V_traj_s=train_V_traj_s, train_V_traj_a=train_V_traj_a,
                                                                   W=W, inner_init_lr=inner_init_lr,
                                                                   b=self.r, gt_advantage_flag=gt_advantage_flag)
                            # Add pi to B
                            valid_pi_list.append(pi)
                        else:
                            # Update using y_star constraint
                            pi_logit -= lr * self.sample_based_NPG(pi=pi, train_Q_traj_s=train_Q_traj_s, train_Q_traj_a=train_Q_traj_a,
                                                                   train_V_traj_s=train_V_traj_s, train_V_traj_a=train_V_traj_a,
                                                                   W=W, inner_init_lr=inner_init_lr,
                                                                   b=self.c(y_set[y_star_idx]), gt_advantage_flag=gt_advantage_flag)

                        end_time = time.time()
                        total_time += (end_time - start_time)

                        del train_Q_traj_s, train_Q_traj_a, train_V_traj_s, train_V_traj_a
                        if self.empty_cache_flag:
                            torch.cuda.empty_cache()

                        # pi is not update yet here.
                        if log_evaluate_flag:
                            true_Obj_array[i] = self.Obj_pi(pi)
                            _, true_max_violat_array[i] = self.check_pi_feasible_true_P(pi, check_fineness)
                    except torch.cuda.OutOfMemoryError as oom_error:
                        warnings.warn(f'Warning: catch OOM error: {oom_error}')
                        if self.empty_cache_flag:
                            raise RuntimeError('Out of memory even if releasing the memory!')
                        self.empty_cache_flag = True
                        pi_logit = old_pi_logit
                        if valid_pi_list_update_flag:
                            valid_pi_list.pop()
                        continue
                    log(f'Iter {i}/{iter_upper_bound}: True Obj: {true_Obj_array[i]}, True Max Violate: {true_max_violat_array[i]}, '
                        f'Sample Max Violate: {max_violate} '
                        f'Time: {end_time - start_time}s, Total Time: {total_time}s, Valid size: {len(valid_pi_list)}',
                        logfile, silent_flag)
                    self.empty_cache_flag = False
                    pbar.update(1)
                    break
        if len(valid_pi_list) == 0:
            final_pi = torch_softmax(pi_logit)
        else:
            if pi_mode == 'mean':
                final_pi = torch.mean(torch.tensor(valid_pi_list, device=self.device), dim=0)
            elif pi_mode == 'last':
                final_pi = valid_pi_list[-1]
            else:
                final_pi = torch.mean(torch.stack(valid_pi_list[-int(pi_mode):]), dim=0)
        return final_pi, true_Obj_array, true_max_violat_array




