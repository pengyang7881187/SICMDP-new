import os
import torch
import argparse
import numpy as np

from tabular_envs.config import *
from tabular_envs.pollution_env import RandomComplexPollutionEnv
from utils.utility import log, set_seed_and_get_rng, get_datetime_str, get_device


parser = argparse.ArgumentParser(description='Hyperparameters of the experiment of comparing with baseline on tabular pollution envs')
parser.add_argument('-n', '--name', default='tabular_SICPO', type=str)
parser.add_argument('--gamma', default=GAMMA, type=int)
parser.add_argument('--dim_Y', default=2, type=int)
parser.add_argument('--S', default=8, type=int)
parser.add_argument('--A', default=4, type=int)
parser.add_argument('--check_fineness', default=CHECK_FINENESS, type=int)  # Used for evaluation.
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--epsilon', default=0.015, type=float)
parser.add_argument('--pos_per_state', default=1, type=int)
parser.add_argument('--coeff', default=1 + 1e-6, type=float)

parser.add_argument('--silent_flag', default=0, type=int)

parser.add_argument('-g', '--gpu', default=3, type=int)  # -1 for cpu.

# 算法相关参数
parser.add_argument('--iter_upper_bound', default=ITER_UPPER_BOUND, type=int)
parser.add_argument('--lr', default=(LR_COEFF / sqrt(ITER_UPPER_BOUND)), type=float)
parser.add_argument('--eta', default=ETA, type=float)
parser.add_argument('--y_size', default=SAMPLE_Y_SIZE, type=int)
parser.add_argument('--traj_num', default=TRAJ_NUM, type=int)
parser.add_argument('--traj_len', default=TRAJ_LEN, type=int)
parser.add_argument('--train_traj_num', default=TRAIN_TRAJ_NUM, type=int)
parser.add_argument('--train_traj_len', default=TRAIN_TRAJ_LEN, type=int)
parser.add_argument('--inner_init_lr', default=INNER_INIT_LR, type=float)
parser.add_argument('--W', default=DEFAULT_W, type=float)
parser.add_argument('--pi_mode', default=PI_MODE, type=str)

parser.add_argument('--seed', default=SEED, type=int)

parser.add_argument(
    '--optimize_y', action='store_true',
    help='Whether to use optimization.'
)

args = parser.parse_args()

name = args.name

gamma = args.gamma
dim_Y = args.dim_Y
S = args.S
A = args.A
check_fineness = args.check_fineness
repeat_time = args.repeat
epsilon = args.epsilon
pos_per_state = args.pos_per_state
coeff = args.coeff

silent_flag = bool(args.silent_flag)

gpu = args.gpu
device = get_device(gpu)

iter_upper_bound = args.iter_upper_bound
lr = args.lr
eta = args.eta
y_size = args.y_size
traj_num = args.traj_num
traj_len = args.traj_len
train_traj_num = args.train_traj_num
train_traj_len = args.train_traj_len
inner_init_lr = args.inner_init_lr
pi_mode = args.pi_mode

W = args.W * S * A

seed = args.seed

rng = set_seed_and_get_rng(seed)

time_str = get_datetime_str()

current_dir = os.path.join(working_dir, name, time_str)
if not os.path.exists(current_dir):
    os.makedirs(current_dir)
logfile = os.path.join(current_dir, 'log.txt')

baseline_grid_fineness_lst = [15, 22, 32]
baseline_grid_num_lst = np.array([250, 500, 1000])
baseline_traj_num_lst = np.array([traj_num] * len(baseline_grid_fineness_lst))

k = 0
while True:
    if k >= repeat_time:
        break
    torch.cuda.empty_cache()
    env = RandomComplexPollutionEnv(S=S, A=A, pos_per_state=pos_per_state, dim_Y=dim_Y, coeff=coeff,
                                    gamma=gamma, device=device)
    P, r, state_coordinates = env.save()

    _, _, _, _, true_max_cons_violat, true_Obj = env.SI_plan(iter_upper_bound, check_fineness, check_fineness,
                                                             silent_flag=True)
    torch.cuda.empty_cache()
    # SICPO
    pi, SICPO_Obj_array, SICPO_max_violat_array = \
        env.SICPO(exp_name=f'Exp_{k}', log_dir=current_dir, eta=eta, lr=lr, silent_flag=True,
                  iter_upper_bound=iter_upper_bound, y_size=y_size, traj_num=traj_num, traj_len=traj_len,
                  train_traj_num=train_traj_num, train_traj_len=train_traj_len, inner_init_lr=inner_init_lr, W=W,
                  pi_mode=pi_mode, log_evaluate_flag=True, check_fineness=check_fineness,
                  optimize_y_flag=args.optimize_y,
                  )
    final_Obj = env.Obj_pi(pi)
    _, final_max_cons_violat = env.check_pi_feasible_true_P(pi, check_fineness)

    final_Obj_gap = max(true_Obj - final_Obj, 0.)
    final_max_cons_violat_gap = max(final_max_cons_violat, 0.) - max(true_max_cons_violat, 0.)

    # CRPO on grids
    for idx, grid_fineness in enumerate(baseline_grid_fineness_lst):
        torch.cuda.empty_cache()
        # To avoid cuda out of memory, you can set gt_evaluate_flag=True in baseline.
        gt_evaluate_flag = False
        if grid_fineness > 20:
            gt_evaluate_flag = True
        pi_baseline, CRPO_Obj_array, CRPO_max_violat_array = \
            env.CRPO(exp_name=f'Exp_{k}', log_dir=current_dir, eta=eta, lr=lr, silent_flag=True,
                     iter_upper_bound=iter_upper_bound, grid_fineness=grid_fineness, traj_num=baseline_traj_num_lst[idx],
                     traj_len=traj_len,
                     train_traj_num=train_traj_num, train_traj_len=train_traj_len,
                     inner_init_lr=inner_init_lr, W=W,
                     pi_mode=pi_mode, log_evaluate_flag=True, check_fineness=check_fineness,
                     gt_evaluate_flag=gt_evaluate_flag
                     )
        final_CRPO_Obj = env.Obj_pi(pi_baseline)
        _, final_CRPO_max_cons_violat = env.check_pi_feasible_true_P(pi_baseline, check_fineness)

        final_CRPO_Obj_gap = max(true_Obj - final_CRPO_Obj, 0.)
        final_CRPO_max_cons_violat_gap = max(final_CRPO_max_cons_violat, 0.) - max(true_max_cons_violat, 0.)

        log(f'CRPO with fineness {grid_fineness}: Obj gap: {final_CRPO_Obj_gap}, Max vio gap: {final_CRPO_max_cons_violat_gap}', logfile, silent_flag)

        k += 1
