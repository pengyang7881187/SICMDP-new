import numpy as np
import torch

from torch import Tensor
from typing import Tuple
from torch.linalg import norm, solve

from .SICMDP import SICMDPEnv


class RandomComplexPollutionEnv(SICMDPEnv):
    def __init__(self, device: torch.device, S=4, A=4, pos_per_state=10, dim_Y=2, coeff=1+1e-6, gamma=0.9,
                 tgt_mode='Action_uniform'):
        name = 'random complex pollution envs'

        P = torch.rand(S, A, S).to(device)
        P = P / (torch.sum(P, dim=2)[:, :, None])

        r = torch.rand(S, A).to(device)

        self.state_coordinates = torch.rand((S, pos_per_state, dim_Y), device=device) * 2.

        def f(x):
            return 1. / (1. + x * x)

        def f_deri(x):
            return - (2. * x) / ((1. + x * x) ** 2.)

        def f_hess(x):
            return ((8. * (x * x)) / ((1. + x * x) ** 3.)) - (2. / ((1. + x * x) ** 2.))

        def c(y: Tensor) -> Tensor:
            if y.dim() == 1 and len(y) == dim_Y:
                c_s_per_pos = f(norm(y - self.state_coordinates, dim=2))  # (S, pos_per_state)
                c_s = torch.sum(c_s_per_pos, dim=1)[:, None]  # (S, 1)
                return c_s.expand(S, A)  # (S, A)
            elif y.dim() == 2:
                y_num = y.shape[0]
                y_axis = torch.swapaxes(y[:, :, None, None], 1, 3)  # (y_num, 1, 1, dim_Y)
                c_s_per_pos = f(norm(y_axis - self.state_coordinates, axis=3))  # (y_num, S, pos_per_state)
                c_s = torch.sum(c_s_per_pos, dim=2)[:, :, None]  # (y_num, S, 1)
                return c_s.expand(y_num, S, A)
            # Unexpected behavior
            else:
                raise ValueError

        def c_grad(y: Tensor) -> Tensor:
            if y.dim() == 1 and len(y) == dim_Y:
                y_minus_state = y - self.state_coordinates  # Shape (S, pos_per_state, dim_Y)
                y_pollution_pos_distant = norm(y_minus_state, dim=2)  # Shape (S, pos_per_state)
                # Shape (S, dim_Y)
                c_grad_without_A = torch.sum(
                    (f_deri(y_pollution_pos_distant) / y_pollution_pos_distant)[:, :, None] * y_minus_state,
                    dim=1
                )
                # Return shape (S, A, dim_Y)
                return c_grad_without_A[:, None, :].repeat(1, A, 1)
            else:
                raise NotImplementedError

        def c_hess(y: Tensor) -> Tensor:
            if y.dim() == 1 and len(y) == dim_Y:
                y_minus_state = y - self.state_coordinates  # Shape (S, pos_per_state, dim_Y)
                y_pollution_pos_distant = norm(y_minus_state, dim=2)  # Shape (S, pos_per_state)
                # Shape (S, pos_per_state, dim_Y)
                partial_d_y_state_to_partial_y = y_minus_state / y_pollution_pos_distant[:, :, None]
                # Shape (S, pos_per_state, dim_Y, dim_Y)
                partial_d_y_state_to_partial_y_outer = torch.einsum('spij, spjk->spik',
                                                                    partial_d_y_state_to_partial_y[:, :, :, None],
                                                                    partial_d_y_state_to_partial_y[:, :, None, :])
                identity = torch.eye(n=dim_Y, device=device)

                # Shape (S, pos_per_state, dim_Y, dim_Y)
                partial2_d_y_state_to_partial_y2 = \
                    (identity - partial_d_y_state_to_partial_y_outer) / y_pollution_pos_distant[:, :, None, None]

                # Shape (S, dim_Y, dim_Y)
                c_hess_without_A = torch.sum(
                    f_hess(y_pollution_pos_distant)[:, :, None, None] * partial_d_y_state_to_partial_y_outer + \
                    f_deri(y_pollution_pos_distant)[:, :, None, None] * partial2_d_y_state_to_partial_y2,
                    dim=1
                )

                # Return shape (S, A, dim_Y)
                return c_hess_without_A[:, None, :, :].repeat(1, A, 1, 1)
            else:
                raise NotImplementedError


        mu = torch.ones((S), device=device) / S
        if tgt_mode == 'Action_uniform':
            tgt_pi = torch.ones((S, A), device=device) / A
            tgt_pi_axis = tgt_pi[:, :, None]
            tgt_P_pi = torch.sum(P * tgt_pi_axis, dim=1)
            d_target = solve((torch.eye(S, device=device) - gamma * tgt_P_pi).T, mu) * (1 - gamma)
        elif tgt_mode == 'Uniform':
            d_target = torch.ones((S), device=device) / S
        else:
            raise

        def u(y: Tensor) -> Tensor:
            if y.dim() == 1 and len(y) == dim_Y:
                return coeff * (1. / (1 - gamma)) * torch.sum(c(y)[:, 0] * d_target)
            elif y.dim() == 2:
                c_array = c(y)
                return coeff * (1. / (1 - gamma)) * torch.sum(c_array[:, :, 0] * d_target, dim=1)
            # Unexpected behavior
            else:
                raise ValueError

        def u_grad(y: Tensor) -> Tensor:
            if y.dim() == 1 and len(y) == dim_Y:
                return coeff * (1. / (1 - gamma)) * torch.sum(c_grad(y)[:, 0, :] * d_target[:, None], dim=0)
            else:
                raise NotImplementedError

        def u_hess(y: Tensor) -> Tensor:
            if y.dim() == 1 and len(y) == dim_Y:
                return coeff * (1. / (1 - gamma)) * torch.sum(c_hess(y)[:, 0, :, :] * d_target[:, None, None], dim=0)
            else:
                raise NotImplementedError

        lb_Y = torch.zeros((dim_Y), device=device)
        ub_Y = torch.ones((dim_Y), device=device) * 2.
        super(RandomComplexPollutionEnv, self).__init__(name=name, S=S, A=A, gamma=gamma, P=P, r=r, c=c, u=u,
                                                        c_grad=c_grad, u_grad=u_grad, c_hess=c_hess, u_hess=u_hess,
                                                        mu=mu, dim_Y=dim_Y, lb_Y=lb_Y, ub_Y=ub_Y, device=device)

    def save(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.P.cpu().numpy(), self.r.cpu().numpy(), self.state_coordinates.cpu().numpy())

    def load(self, P: Tensor, r: Tensor, state_coordinates: Tensor):
        self.P = P.to(self.device)
        self.r = r.to(self.device)
        self.state_coordinates = state_coordinates.to(self.device)
        return

