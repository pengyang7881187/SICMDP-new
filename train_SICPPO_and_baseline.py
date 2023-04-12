import os
import ray
import tempfile
import argparse
import numpy as np

from pprint import pprint

from ray.tune.logger import UnifiedLogger

from utils import filter_result
from eval import eval_pollution_env_function
from envs.pollution_env import PollutionEnv
from algorithms.sicppo import SICPPOConfig
from algorithms.sicppo_policy import SICPPOPolicy
from configs import DEFAULT_POLLUTION_CONFIG, SAVE_CHECKPOINT_DIR, RAY_RESULTS_DIR
from utils.utility import (
    get_datetime_str,
    update_env_config
)


from configs.pollution_env_config import MAX_STEPS

from find_feasible_pollution_env import get_monomial_func


class StopCriterion:
    def __init__(self, episode_len_mean_ub: float, train_max_violat_ub: float, reward_tolerance: int):
        self.episode_len_mean_ub = episode_len_mean_ub
        self.train_max_violat_ub = train_max_violat_ub
        self.reward_tolerance = reward_tolerance
        self.best_reward = -np.inf
        self.reward_no_increase = 0
        return

    def ifstop_and_update(self, episode_len_mean: float, train_max_violat: float, reward: float) -> bool:
        if episode_len_mean <= self.episode_len_mean_ub and train_max_violat <= self.train_max_violat_ub:
            self.best_reward = max(self.best_reward, reward)
            if reward < self.best_reward - 1e-6:
                self.reward_no_increase += 1
            else:
                self.reward_no_increase = 0
            if self.reward_no_increase == self.reward_tolerance:
                self.reset()
                return True
        else:
            return False

    def reset(self):
        self.best_reward = -np.inf
        self.reward_no_increase = 0
        return


def custom_log_creator(custom_str: str, custom_path: str = RAY_RESULTS_DIR):
    logdir_prefix = f'{custom_str}_'
    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


# os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'


parser = argparse.ArgumentParser(description='Hyperparameters of the experiment of deep pollution environment.')

# Basic arguments.
parser.add_argument(
    '-n', '--name', default='long_lr1e4', type=str,
    help='Name of experiment.'
)
parser.add_argument(
    '--num_seeds', default=20, type=int,
    help='Number of random seeds.'
)
parser.add_argument(
    '-s', '--seed', default=12, type=int,
    help='Start random seed.'
)
parser.add_argument(
    '--gamma', default=1., type=float,
    help='Discount rate.'
)

# Environment.
parser.add_argument(
    '--env_seed', default=3, type=int,
    help='Random seed for environment.'
)
parser.add_argument(
    '--env_max_steps', default=MAX_STEPS, type=int,
    help='Max steps for environment.'
)
parser.add_argument(
    '--env_pollution_func_name', default='exponential', type=str,
    choices=('exponential', 'reciprocal'),
    help='Pollution function name for environment.'
)
parser.add_argument(
    '--env_pollution_func_parameter', default=20., type=float,
    help='Pollution function parameter for environment.'
)
parser.add_argument(
    '--env_protect_func_name', default='exponential', type=str,
    choices=('exponential', 'monomial'),
    help='Protect function name for environment.'
)
parser.add_argument(
    '--env_protect_func_parameter', default=20., type=float,
    help='Protect function parameter for environment.'
)
parser.add_argument(
    '--env_protect_func_multiplier', default=0.005, type=float,
    help='Protect function multiplier for environment.'
)
parser.add_argument(
    '--env_protect_func_intercept', default=0.02, type=float,
    help='Protect function intercept for environment.'
)
parser.add_argument(
    '--constant_speed', action='store_true',
    help='Whether to use constant speed in the environment.'
)
parser.add_argument(
    '--symmetric', action='store_true',
    help='Whether to utilize symmetric in the environment.'
)


# Model.
parser.add_argument(
    '--activation', default='tanh', type=str,
    help='Activation function.'
)
parser.add_argument(
    '--action_dist', default=None, type=str,
    help='Activation function.'
)  # The action distribution must be in register.
parser.add_argument(
    '--num_layers', default=2, type=int,
    help='Hidden layers of the model.'
)
parser.add_argument(
    '--hidden_size', default=512, type=int,
    help='Hidden of each layer of the model.'
)
parser.add_argument(
    '--not_use_critic', action='store_true',
    help='Whether to use critic.'
)
parser.add_argument(
    '--not_use_gae', action='store_true',
    help='Whether to use gae.'
)

# Training.
parser.add_argument(
    '--lr', default=1e-4, type=float,
    help='Learning rate.'
)
parser.add_argument(
    '--num_y', default=100, type=int,
    help='Number of constraint points y sampled in SICPPO.'
)
parser.add_argument(
    '--eta', default=0.01, type=float,
    help='Tolerance of constraint violation.'
)
parser.add_argument(
    '--sgd_minibatch_size', default=800, type=int,
    help='Minibatch size of each SGD iteration.'
)
parser.add_argument(
    '--num_sgd_iter', default=10, type=int,
    help='Number of SGD iterations for each training batch.'
)
parser.add_argument(
    '--kl_coeff', default=0.05, type=float,
    help='KL coefficient in the loss.'
)
parser.add_argument(
    '--vf_loss_coeff', default=0.1, type=float,
    help='Value function coefficient in the loss.'
)
parser.add_argument(
    '--constraint_vf_loss_coeff', default=0.1, type=float,
    help='Constraint value function coefficient in the loss.'
)
parser.add_argument(
    '--max_constraint_vf_loss_coeff', default=0.1, type=float,
    help='Max violate constraint value function coefficient in the loss.'
)
parser.add_argument(
    '--policy_loss_coeff', default=1., type=float,
    help='Policy coefficient in the loss.'
)
parser.add_argument(
    '--entropy_coeff', default=0., type=float,
    help='Entropy coefficient in the loss.'
)
parser.add_argument(
    '--clip_param', default=0.3, type=float,
    help='Clip parameter for likelihood ratio in PPO loss.'
)
parser.add_argument(
    '--vf_clip_param', default=10., type=float,
    help='Clip parameter for value function loss.'
)
parser.add_argument(
    '--constraint_vf_clip_param', default=10., type=float,
    help='Clip parameter for constraint value function loss.'
)
parser.add_argument(
    '--grad_clip', default=None, type=float,
    help='Gradient clip parameter of total loss.'
)
parser.add_argument(
    '--kl_target', default=0.01, type=float,
    help='Target value for KL divergence.'
)
parser.add_argument(
    "--not_shuffle", action='store_true',
    help="Whether to shuffle the training batch while training."
)

# Stop criterion.
parser.add_argument(
    "--stop_iters", type=int, default=400,
    help="Number of iterations to train."
)
parser.add_argument(
    "--stop_reward_no_increase_tolerance", type=int, default=3,
    help="Number of iterations to wait for increasing of reward."
)
parser.add_argument(
    "--stop_episode_len_mean_ub", type=float, default=20.,
    help="When episode len mean is lower than this upper bound, we consider stopping the algorithm."
)

# Evaluation.
parser.add_argument(
    "--evaluation_interval", type=int, default=2,
    help="Evaluate with every evaluation_interval training iterations."
)
parser.add_argument(
    "--evaluation_duration", type=int, default=1,
    help="Duration for which to run evaluation each evaluation_interval."
)
parser.add_argument(
    "--evaluation_num_workers", type=int, default=0,
    help="Number of parallel workers to use for evaluation."
)

# Resource.
parser.add_argument(
    "--num_gpus", type=int, default=4,
    help="Number of allocated gpus"
)

# Rollout worker.
parser.add_argument(
    "--num_rollout_workers", type=int, default=4,
    help="Number of rollout workers."
)
parser.add_argument(
    "--num_envs_per_rollout_worker", type=int, default=1,
    help="Number of environment for each rollout worker."
)
parser.add_argument(
    "--remote_worker_envs", action='store_true',
    help="Whether to use additional remote workers for each rollout worker."
)
parser.add_argument(
    "--rollout_batch_mode", type=str, default='complete_episodes', choices=('complete_episodes', 'truncated_episodes'),
    help="Mode of sampling for rollout workers."
)
# We collect at least 10 episodes for each worker.
parser.add_argument(
    "--rollout_fragment_length", type=int, default=16 * MAX_STEPS,
    help="Number of steps collected by each rollout worker."
)
parser.add_argument(
    "--num_cpus_per_rollout_worker", type=int, default=1,
    help="Number of cpus allocated for each rollout worker."
)

# Trainer worker.
parser.add_argument(
    "--save_checkpoint_interval", type=int, default=30,
    help="Save checkpoint every evaluation_interval training iterations."
)
parser.add_argument(
    "--num_trainer_workers", type=int, default=4,
    help="Number of trainer workers."
)
parser.add_argument(
    "--num_cpus_per_trainer_worker", type=int, default=1,
    help="Number of cpus allocated for each trainer workers."
)

# Checkpoint.
parser.add_argument(
    '--train_from_checkpoint_flag', action='store_true',
    help='Whether to use checkpoint.'
)
parser.add_argument(
    '--load_checkpoint_path', type=str,
    help='Relative path of the checkpoint.'
)

# Present setting.
parser.add_argument(
    '--silent', action='store_true',
    help='Whether to print during logging.'
)
parser.add_argument(
    '--render', action='store_true',
    help='Whether to render during evaluating.'
)

if __name__ == "__main__":
    args = parser.parse_args()
    assert args.gamma <= 1.

    experiment_setting_dict = vars(args)
    pprint(experiment_setting_dict)

    # Set name.
    datetime_str = get_datetime_str()
    name = args.name + '_' + datetime_str

    silent_flag = args.silent

    ray.init()

    env_config = DEFAULT_POLLUTION_CONFIG.copy()
    update_env_config(env_config, key='seed', val=args.env_seed)
    update_env_config(env_config, key='max_steps', val=args.env_max_steps)
    update_env_config(env_config, key='pollution_func_name', val=args.env_pollution_func_name)
    update_env_config(env_config, key='protect_func_name', val=args.env_protect_func_name)
    update_env_config(env_config, key='pollution_func_parameter', val=args.env_pollution_func_parameter)
    update_env_config(env_config, key='protect_func_parameter', val=args.env_protect_func_parameter)
    update_env_config(env_config, key='protect_func_multiplier', val=args.env_protect_func_multiplier)
    update_env_config(env_config, key='protect_func_intercept', val=args.env_protect_func_intercept)
    # update_env_config(env_config, key='constant_speed_flag', val=args.constant_speed)
    update_env_config(env_config, key='constant_speed_flag', val=True)
    # update_env_config(env_config, key='symmetric_flag', val=args.symmetric)
    update_env_config(env_config, key='symmetric_flag', val=True)

    optimize_y_flag = True

    sicppo_config = (
        SICPPOConfig()
        .environment(
            env=PollutionEnv,
            env_config=env_config,
            # render_env=args.render
        )
        .framework('torch')
        .rollouts(
            num_rollout_workers=args.num_rollout_workers,
            num_envs_per_worker=args.num_envs_per_rollout_worker,
            remote_worker_envs=args.remote_worker_envs,
            batch_mode=args.rollout_batch_mode,
            rollout_fragment_length=args.rollout_fragment_length,
            recreate_failed_workers=True,
        )
        .evaluation(
            custom_evaluation_function=eval_pollution_env_function,
            evaluation_config={
                'explore': True,
                # 'render_flag': args.render
                'render_flag': True,
                'predefined_path': PollutionEnv(DEFAULT_POLLUTION_CONFIG).discretize_path(*get_monomial_func(4)),
                'eval_optimize_y_flag': True
            },
            evaluation_interval=args.evaluation_interval,
            evaluation_duration_unit='episodes',
            evaluation_duration=args.evaluation_duration,
            evaluation_num_workers=args.evaluation_num_workers
        )
        .training(
            gamma=args.gamma,
            lr=args.lr,
            num_y=args.num_y,
            eta=args.eta,
            # lr_schedule=None,
            train_batch_size=args.rollout_fragment_length * args.num_rollout_workers * args.num_envs_per_rollout_worker,
            use_critic=not args.not_use_critic,
            use_gae=not args.not_use_gae,
            kl_coeff=args.kl_coeff,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            shuffle_sequences=not args.not_shuffle,
            vf_loss_coeff=args.vf_loss_coeff,
            constraint_vf_loss_coeff=args.constraint_vf_loss_coeff,
            entropy_coeff=args.entropy_coeff,
            # entropy_coeff_schedule=None,
            clip_param=args.clip_param,
            vf_clip_param=args.vf_clip_param,
            constraint_vf_clip_param=args.constraint_vf_clip_param,
            grad_clip=args.grad_clip,
            kl_target=args.kl_target,
            optimize_y_flag=optimize_y_flag,
            max_constraint_vf_loss_coeff=args.max_constraint_vf_loss_coeff,
            policy_loss_coeff=args.policy_loss_coeff,
            # optimizer=None,
            model={
                "custom_action_dist": args.action_dist,
                "vf_share_layers": False,
                "fcnet_hiddens": [args.hidden_size] * args.num_layers,
                "fcnet_activation": args.activation,
                "custom_model": "sicppo_fc_model",
                "custom_model_config": {
                    'dim_Y': 2
                }
            },
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` envs var set to > 0.
        .resources(
            num_gpus=args.num_gpus,
            num_cpus_per_worker=args.num_cpus_per_rollout_worker,
            num_gpus_per_worker=args.num_gpus / args.num_rollout_workers,
            num_trainer_workers=args.num_trainer_workers,
            num_cpus_per_trainer_worker=args.num_cpus_per_trainer_worker,
            num_gpus_per_trainer_worker=args.num_gpus / args.num_trainer_workers,
        )
    )
    seed_lst = list(range(args.seed, args.seed+args.num_seeds))

    crpo_grid_fineness_lst = [15, 22, 32]
    crpo_config_lst = [sicppo_config.copy().training(crpo_flag=True, crpo_grid_fineness=grid_fineness,
                                                     optimize_y_flag=False)
                       for grid_fineness in crpo_grid_fineness_lst]

    stop_criterion = StopCriterion(episode_len_mean_ub=args.stop_episode_len_mean_ub, train_max_violat_ub=args.eta,
                                   reward_tolerance=args.stop_reward_no_increase_tolerance)

    for seed in seed_lst:
        sicppo_config.debugging(
            seed=seed,
            logger_creator=custom_log_creator(custom_str=f'{name}_SICPPO_seed_{seed}')
        )
        sicppo_algo = sicppo_config.build()
        if args.train_from_checkpoint_flag:
            sicppo_algo.get_policy().set_weights(
                SICPPOPolicy.from_checkpoint(
                    checkpoint=os.path.join(SAVE_CHECKPOINT_DIR,
                                            args.load_checkpoint_path,
                                            # load_checkpoint_path example:
                                            # long_lr1e4_2023-03-29-13:01:09/checkpoint_000141
                                            'policies/default_policy'),
                ).get_weights()
            )
            sicppo_algo.workers.sync_weights()
        for iter_num in range(args.stop_iters):
            result = sicppo_algo.train()
            filtered_result = filter_result(result)
            print(f'SICPPO: seed-{seed}')
            pprint(filtered_result)
            if stop_criterion.ifstop_and_update(episode_len_mean=filtered_result['episode_len_mean'],
                                                train_max_violat=filtered_result['train_max_constraint_violat'],
                                                reward=filtered_result['episode_reward_mean']):
                sicppo_algo.save(checkpoint_dir=os.path.join(SAVE_CHECKPOINT_DIR, f'{name}_seed_{seed}'))
                break
            if iter_num % args.save_checkpoint_interval == 0:
                sicppo_algo.save(checkpoint_dir=os.path.join(SAVE_CHECKPOINT_DIR, f'{name}_seed_{seed}'))
        sicppo_algo.stop()

        # Baseline
        for crpo_idx, crpo_config in enumerate(crpo_config_lst):
            crpo_config.debugging(
                seed=seed,
                logger_creator=custom_log_creator(custom_str=f'{name}_CRPO_{crpo_grid_fineness_lst[crpo_idx]}_seed_{seed}')
            )
            crpo_algo = crpo_config.build()
            crpo_name = f'{name}_crpo_{crpo_grid_fineness_lst[crpo_idx]}'
            for iter_num in range(args.stop_iters):
                result = crpo_algo.train()
                filtered_result = filter_result(result)
                print(f'CRPO-{crpo_grid_fineness_lst[crpo_idx]}: seed-{seed}')
                pprint(filtered_result)
                if stop_criterion.ifstop_and_update(episode_len_mean=filtered_result['episode_len_mean'],
                                                    train_max_violat=filtered_result['train_max_constraint_violat'],
                                                    reward=filtered_result['episode_reward_mean']):
                    crpo_algo.save(checkpoint_dir=os.path.join(SAVE_CHECKPOINT_DIR, f'{crpo_name}_seed_{seed}'))
                    break
                if iter_num % args.save_checkpoint_interval == 0:
                    crpo_algo.save(checkpoint_dir=os.path.join(SAVE_CHECKPOINT_DIR, f'{crpo_name}_seed_{seed}'))
            crpo_algo.stop()
    ray.shutdown()
