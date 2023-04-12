"""
Semi-Infinitely Constrained Proximal Policy Optimization (SICPPO)
==================================

This file defines the distributed Algorithm class for semi-infinitely constrained proximal policy optimization.

The code is modified from https://docs.ray.io/en/latest/_modules/ray/rllib/algorithms/ppo/ppo.html.
"""
import torch
import logging
import warnings
import numpy as np

from numpy import ndarray
from scipy.optimize import minimize
from typing import List, Optional, Type, Union, Tuple, TYPE_CHECKING

from ray.util.debug import log_once
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
    DEPRECATED_VALUE,
    deprecation_warning,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)

if TYPE_CHECKING:
    from ray.rllib.core.rl_module import RLModule

from .rollout_ops import (
    synchronous_parallel_sample_and_process_constraints,
    SICMDPRolloutWorker
)

from envs import SICMDPEnv
from models import SICPPOFCNet
from utils import DEFAULT_POLICY_IDX, ArrayType

from .sicppo_policy import SICPPOPolicy

logger = logging.getLogger(__name__)


ModelCatalog.register_custom_model(
    "sicppo_fc_model", SICPPOFCNet
)


class OptimizeError(Exception):
    def __init__(self, message='Fail to optimize.'):
        super(OptimizeError, self).__init__(message)
        return


class SICPPOConfig(PGConfig):
    """Defines a configuration class from which a SICPPO Algorithm can be built.

    Example:
        >>> from algorithms.sicppo import SICPPOConfig
        >>> config = SICPPOConfig()  # doctest: +SKIP
        >>> config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)  # doctest: +SKIP
        >>> config = config.resources(num_gpus=0)  # doctest: +SKIP
        >>> config = config.rollouts(num_rollout_workers=4)  # doctest: +SKIP
        >>> print(config.to_dict())  # doctest: +SKIP
        >>> # Build an Algorithm object from the config and run 1 training iteration.
        >>> algo = config.build(envs="CartPole-v1")  # doctest: +SKIP
        >>> algo.train()  # doctest: +SKIP

    Example:
        >>> from algorithms.sicppo import SICPPOConfig
        >>> from ray import air
        >>> from ray import tune
        >>> config = SICPPOConfig()
        >>> # Print out some default values.
        >>> print(config.clip_param)  # doctest: +SKIP
        >>> # Update the config object.
        >>> config.training(  # doctest: +SKIP
        ... lr=tune.grid_search([0.001, 0.0001]), clip_param=0.2
        ... )
        >>> # Set the config object's envs.
        >>> config = config.environment(envs="CartPole-v1")   # doctest: +SKIP
        >>> # Use to_dict() to get the old-style python config dict
        >>> # when running with tune.
        >>> tune.Tuner(  # doctest: +SKIP
        ...     "SICPPO",
        ...     run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
        ...     param_space=config.to_dict(),
        ... ).fit()
    """

    def __init__(self, algo_class=None):
        """Initializes a SICPPOConfig instance."""
        super().__init__(algo_class=algo_class or SICPPO)
        # SICPPO specific settings:
        self.worker_cls = SICMDPRolloutWorker
        self.num_y = 100  # The key is not used if crpo_flag is True.
        self.eta = 0.2  # Constraint violate criterion.
        self.constraint_vf_clip_param = 10.
        self.constraint_vf_loss_coeff = 1.0
        self.crpo_flag = False
        self.crpo_grid_fineness = 32
        self.optimize_y_flag = False

        self.max_constraint_vf_loss_coeff = 1.0

        # fmt: off
        # __sphinx_doc_begin__
        # PPO settings:
        self.use_critic = True
        self.use_gae = True
        self.lambda_ = 1.0
        self.kl_coeff = 0.2
        self.sgd_minibatch_size = 400
        self.num_sgd_iter = 30
        self.shuffle_sequences = True
        self.policy_loss_coeff = 1.
        self.vf_loss_coeff = 1.0
        self.entropy_coeff = 0.0
        self.entropy_coeff_schedule = None
        self.clip_param = 0.3
        self.vf_clip_param = 10.0
        self.grad_clip = None
        self.kl_target = 0.01

        # Override some of PG/AlgorithmConfig's default values with SICPPO-specific values.
        self.num_rollout_workers = 8
        self.train_batch_size = 8000
        self.lr = 5e-5
        self.model["vf_share_layers"] = False  # share layers for value function (critic)
        self._disable_preprocessor_api = False
        # __sphinx_doc_end__
        # fmt: on

        # Deprecated keys.
        self.vf_share_layers = DEPRECATED_VALUE

    @override(AlgorithmConfig)
    def get_default_rl_module_class(self) -> Union[Type["RLModule"], str]:
        raise NotImplementedError('_enable_rl_trainer_api is not supported!')

    @override(AlgorithmConfig)
    def training(
            self,
            *,
            lr_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
            use_critic: Optional[bool] = NotProvided,
            use_gae: Optional[bool] = NotProvided,
            lambda_: Optional[float] = NotProvided,
            kl_coeff: Optional[float] = NotProvided,
            sgd_minibatch_size: Optional[int] = NotProvided,
            num_sgd_iter: Optional[int] = NotProvided,
            shuffle_sequences: Optional[bool] = NotProvided,
            policy_loss_coeff: Optional[float] = NotProvided,
            vf_loss_coeff: Optional[float] = NotProvided,
            entropy_coeff: Optional[float] = NotProvided,
            entropy_coeff_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
            clip_param: Optional[float] = NotProvided,
            vf_clip_param: Optional[float] = NotProvided,
            grad_clip: Optional[float] = NotProvided,
            kl_target: Optional[float] = NotProvided,
            # Deprecated.
            vf_share_layers=DEPRECATED_VALUE,
            # SICPPO
            num_y: Optional[int] = NotProvided,
            eta: Optional[float] = NotProvided,
            constraint_vf_clip_param: Optional[float] = NotProvided,
            constraint_vf_loss_coeff: Optional[float] = NotProvided,
            crpo_flag: Optional[bool] = NotProvided,
            crpo_grid_fineness: Optional[int] = NotProvided,
            optimize_y_flag: Optional[bool] = NotProvided,
            max_constraint_vf_loss_coeff: Optional[float] = NotProvided,
            **kwargs,
    ) -> "SICPPOConfig":
        """Sets the training related configuration.

        Args:
            lr_schedule: Learning rate schedule. In the format of
                [[timestep, lr-value], [timestep, lr-value], ...]
                Intermediary timesteps will be assigned to interpolated learning rate
                values. A schedule should normally start from timestep 0.
            use_critic: Should use a critic as a baseline (otherwise don't use value
                baseline; required for using GAE).
            use_gae: If true, use the Generalized Advantage Estimator (GAE)
                with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            lambda_: The GAE (lambda) parameter.
            kl_coeff: Initial coefficient for KL divergence.
            sgd_minibatch_size: Total SGD batch size across all devices for SGD.
                This defines the minibatch size within each epoch.
            num_sgd_iter: Number of SGD iterations in each outer loop (i.e., number of
                epochs to execute per train batch).
            shuffle_sequences: Whether to shuffle sequences in the batch when training
                (recommended).
            vf_loss_coeff: Coefficient of the value function loss. IMPORTANT: you must
                tune this if you set vf_share_layers=True inside your model's config.
            entropy_coeff: Coefficient of the entropy regularizer.
            entropy_coeff_schedule: Decay schedule for the entropy regularizer.
            clip_param: PPO clip parameter.
            vf_clip_param: Clip param for the value function. Note that this is
                sensitive to the scale of the rewards. If your expected V is large,
                increase this.
            grad_clip: If specified, clip the global norm of gradients by this amount.
            kl_target: Target value for KL divergence.

        Returns:
            This updated AlgorithmConfig object.
        """
        if vf_share_layers != DEPRECATED_VALUE:
            deprecation_warning(
                old="ppo.DEFAULT_CONFIG['vf_share_layers']",
                new="PPOConfig().training(model={'vf_share_layers': ...})",
                error=True,
            )

        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if use_critic is not NotProvided:
            self.use_critic = use_critic
        if use_gae is not NotProvided:
            self.use_gae = use_gae
        if lambda_ is not NotProvided:
            self.lambda_ = lambda_
        if kl_coeff is not NotProvided:
            self.kl_coeff = kl_coeff
        if sgd_minibatch_size is not NotProvided:
            self.sgd_minibatch_size = sgd_minibatch_size
        if num_sgd_iter is not NotProvided:
            self.num_sgd_iter = num_sgd_iter
        if shuffle_sequences is not NotProvided:
            self.shuffle_sequences = shuffle_sequences
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if entropy_coeff_schedule is not NotProvided:
            self.entropy_coeff_schedule = entropy_coeff_schedule
        if clip_param is not NotProvided:
            self.clip_param = clip_param
        if vf_clip_param is not NotProvided:
            self.vf_clip_param = vf_clip_param
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if kl_target is not NotProvided:
            self.kl_target = kl_target

        # SICPPO
        if num_y is not NotProvided:
            self.num_y = num_y
        if eta is not NotProvided:
            self.eta = eta
        if constraint_vf_clip_param is not NotProvided:
            self.constraint_vf_clip_param = constraint_vf_clip_param
        if constraint_vf_loss_coeff is not NotProvided:
            self.constraint_vf_loss_coeff = constraint_vf_loss_coeff
        if crpo_flag is not NotProvided:
            self.crpo_flag = crpo_flag
        if crpo_grid_fineness is not NotProvided:
            self.crpo_grid_fineness = crpo_grid_fineness
        if optimize_y_flag is not NotProvided:
            self.optimize_y_flag = optimize_y_flag
        if max_constraint_vf_loss_coeff is not NotProvided:
            self.max_constraint_vf_loss_coeff = max_constraint_vf_loss_coeff
        if policy_loss_coeff is not NotProvided:
            self.policy_loss_coeff = policy_loss_coeff

        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()

        # SICPPO
        assert self.num_y > 0
        assert self.eta > 0
        assert self.constraint_vf_clip_param > 0
        assert self.constraint_vf_loss_coeff > 0
        assert self.crpo_grid_fineness > 0

        # SGD minibatch size must be smaller than train_batch_size (b/c
        # we subsample a batch of `sgd_minibatch_size` from the train-batch for
        # each `num_sgd_iter`).
        # Note: Only check this if `train_batch_size` > 0 (DDPPO sets this
        # to -1 to auto-calculate the actual batch size later).
        if self.sgd_minibatch_size > self.train_batch_size:
            raise ValueError(
                f"`sgd_minibatch_size` ({self.sgd_minibatch_size}) must be <= "
                f"`train_batch_size` ({self.train_batch_size}). In PPO, the train batch"
                f" is be split into {self.sgd_minibatch_size} chunks, each of which is "
                f"iterated over (used for updating the policy) {self.num_sgd_iter} "
                "times."
            )

        # Episodes may only be truncated (and passed into PPO's
        # `postprocessing_fn`), iff generalized advantage estimation is used
        # (value function estimate at end of truncated episode to estimate
        # remaining value).
        if (
                not self.in_evaluation
                and self.batch_mode == "truncate_episodes"
                and not self.use_gae
        ):
            raise ValueError(
                "Episode truncation is not supported without a value "
                "function (to estimate the return at the end of the truncated"
                " trajectory). Consider setting "
                "batch_mode=complete_episodes."
            )

        # Check `entropy_coeff` for correctness.
        if self.entropy_coeff < 0.0:
            raise ValueError("`entropy_coeff` must be >= 0.0")


class SICPPO(Algorithm):
    @override(Algorithm)
    def setup(self, config: AlgorithmConfig) -> None:
        super().setup(config)
        self.env_local: SICMDPEnv = self.env_creator(config['env_config'])

        self.crpo_flag = config['crpo_flag']
        self.crpo_grid_fineness = config['crpo_grid_fineness']

        self.optimize_y_flag = config['optimize_y_flag']

        assert not (self.optimize_y_flag and self.crpo_flag)

        self.init_y_for_optimize = self.env_local.y0.copy()  # Start from center of Y.

        self.num_y = config['num_y']
        self.eta = config['eta']
        self.gamma = config['gamma']

        self.env_max_episode_steps = self.env_local.max_episode_steps

        self.discount_tensor = self.gamma ** torch.arange(self.env_max_episode_steps).cuda()

        # We do not support initial distribution of MDP currently.
        self.init_obs, _ = self.env_local.reset()
        self.env_local.set_seed(seed=config['env_config']['seed'])
        if self.crpo_flag or self.optimize_y_flag:
            self.crpo_batch_y = self.env_local.grid.generate(self.crpo_grid_fineness)
            self.num_y = self.crpo_batch_y.shape[0]
        return

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return SICPPOConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(
            cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            if config._enable_rl_module_api:
                raise NotImplementedError('_enable_rl_trainer_api is not supported!')
            else:
                return SICPPOPolicy
        else:
            raise NotImplementedError('Only torch framework is supported!')

    @ExperimentalAPI
    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        # Sample constraint point y.
        if self.crpo_flag or self.optimize_y_flag:
            batch_y = self.crpo_batch_y
        else:
            batch_y = self.env_local.sample_batch_y(self.num_y)
        # Sample and process constraints.
        # Count steps by agent steps.
        if self.config.count_steps_by == "agent_steps":
            train_batch = synchronous_parallel_sample_and_process_constraints(
                worker_set=self.workers,
                max_agent_steps=self.config.train_batch_size,
                batch_y=batch_y
            )
        # Count steps by envs steps.
        else:
            train_batch = synchronous_parallel_sample_and_process_constraints(
                worker_set=self.workers,
                max_env_steps=self.config.train_batch_size,
                batch_y=batch_y
            )
        train_batch = train_batch.as_multi_agent()

        # Parse additional information about constraints to the trainer.
        local_worker = self.workers.local_worker()
        policy_to_train: SICPPOPolicy = local_worker.get_policy(DEFAULT_POLICY_IDX)
        policy_to_train.batch_y = batch_y

        optimize_fail_flag = False
        try:
            max_violat_val, max_violat_y, all_success_flag = \
                self.get_optimal_max_avg_monte_carlo_constraint_violat_value_and_pos_and_all_success_flag(
                    train_batch=train_batch[DEFAULT_POLICY_IDX]
                )
            print(f'Max optimize violat y {max_violat_y}')
            policy_to_train.optimize_max_violat_val = max_violat_val
            if self.optimize_y_flag:
                policy_to_train.all_success_flag = all_success_flag
                policy_to_train.max_violat_val = max_violat_val
                policy_to_train.max_violat_y = max_violat_y
                policy_to_train.current_optimize_y_flag = True
                train_batch = self.set_constraint_information_for_max_violat_y(
                    policy_to_train=policy_to_train,
                    sample_batch=train_batch,
                    y=max_violat_y
                )
        except OptimizeError as optimize_error:
            warnings.warn(f'Warning: Fail to optimize y with error information: {optimize_error}')
            optimize_fail_flag = True
            self.init_y_for_optimize = self.env_local.y0.copy()
        if (not self.optimize_y_flag) or optimize_fail_flag:
            max_violat_val, max_violat_idx, all_success_flag = \
                self.get_sample_max_avg_monte_carlo_constraint_violat_value_and_idx_and_all_success_flag(
                    train_batch=train_batch[DEFAULT_POLICY_IDX], batch_y=batch_y
                )
            policy_to_train.all_success_flag = all_success_flag
            policy_to_train.max_violat_val = max_violat_val
            policy_to_train.max_violat_idx = max_violat_idx
            policy_to_train.current_optimize_y_flag = False
            train_batch = self.set_constraint_information_for_max_violat_y(
                policy_to_train=policy_to_train,
                sample_batch=train_batch,
                y=batch_y[max_violat_idx]
            )

        batch_size = train_batch.agent_steps()

        # Update steps counters.
        assert train_batch.agent_steps() == train_batch.env_steps(), 'We do not support agent_steps != env_steps!'
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages; note that constraint advantages is standardized while sampling.
        train_batch = standardize_fields(train_batch, ["advantages"])

        # Train.
        if self.config._enable_rl_trainer_api:
            raise NotImplementedError('_enable_rl_trainer_api is not supported!')
        elif self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        policies_to_update = list(train_results.keys())

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        # Update weights after learning on the local worker on all remote workers.
        if self.workers.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                from_worker = None
                self.workers.sync_weights(
                    from_worker=from_worker,
                    policies=list(train_results.keys()),
                    global_vars=global_vars,
                )

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                    self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                    log_once("sicppo_warned_lr_ratio")
                    and self.config.get("model", {}).get("vf_share_layers")
                    and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                    log_once("sicppo_warned_vf_clip")
                    and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results

    # For sample-based (w.r.t. y) SICPPO and CRPO.
    def get_sample_max_avg_monte_carlo_constraint_violat_value_and_idx_and_all_success_flag(
            self, train_batch: SampleBatch, batch_y: np.ndarray
    ) -> Tuple[float, int, bool]:
        # Compute Monte-Carlo constraint value function for each complete episode.
        train_batch_with_constraint = SampleBatch({
            'constraints': train_batch[SICMDPEnv.CONSTRAINTS],
            'eps_id': train_batch['eps_id']
        })
        train_batch_split_by_episode = train_batch_with_constraint.split_by_episode()
        # Compute all fail flag: If every episode fails to reach the goal, update the value function.
        episode_len_lst = [sample_episode.agent_steps() for sample_episode in train_batch_split_by_episode]
        max_episode_len = max(episode_len_lst)
        # all_fail_flag = all([episode_len == self.env_max_episode_steps for episode_len in episode_len_lst])
        all_success_flag = (max_episode_len < self.env_max_episode_steps)

        current_discount_tensor = self.discount_tensor[:max_episode_len]

        # Cast constraint to tensor.
        # Shape (num_episodes, env_max_episode_steps, num_y)
        num_episodes = len(train_batch_split_by_episode)
        num_y = batch_y.shape[0]
        constraint_split_by_episode = torch.zeros((num_episodes, max_episode_len, num_y)).cuda()
        for episode_idx, sample_episode in enumerate(train_batch_split_by_episode):
            episode_len = sample_episode[SICMDPEnv.CONSTRAINTS].shape[0]
            constraint_split_by_episode[episode_idx, :episode_len, :] = torch.as_tensor(
                sample_episode[SICMDPEnv.CONSTRAINTS]
            ).cuda()

        # Shape (num_episodes, num_y)
        constraint_value_split_by_episode = torch.sum(constraint_split_by_episode * current_discount_tensor[None, :, None],
                                                      dim=1)
        # Shape (num_y)
        avg_constraint_value = torch.mean(constraint_value_split_by_episode, dim=0).cpu().numpy()
        # Shape (num_y)
        batch_u = self.env_local.batch_u(batch_y)

        avg_constraint_value_violat = avg_constraint_value - batch_u

        max_avg_constraint_value_violat_idx = int(np.argmax(avg_constraint_value_violat))

        # max_avg_constraint_value_violat_y = batch_y[max_avg_constraint_value_violat_idx]

        max_avg_constraint_value_violat = avg_constraint_value_violat[max_avg_constraint_value_violat_idx]

        return max_avg_constraint_value_violat, max_avg_constraint_value_violat_idx, all_success_flag

    # For optimize-based (w.r.t. y) SICPPO.
    def get_optimal_max_avg_monte_carlo_constraint_violat_value_and_pos_and_all_success_flag(
            self, train_batch: SampleBatch
    ) -> Tuple[float, ndarray, bool]:
        # Compute Monte-Carlo constraint value function for each complete episode.
        train_batch_with_obs_and_actions = SampleBatch({
            'obs': train_batch['obs'],
            'actions': train_batch['actions'],
            'eps_id': train_batch['eps_id']
        })

        train_batch_split_by_episode = train_batch_with_obs_and_actions.split_by_episode()
        # Compute all fail flag: If every episode fails to reach the goal, update the value function.
        episode_len_lst = [sample_episode.agent_steps() for sample_episode in train_batch_split_by_episode]
        max_episode_len = max(episode_len_lst)
        # all_fail_flag = all([episode_len == self.env_max_episode_steps for episode_len in episode_len_lst])
        all_success_flag = (max_episode_len < self.env_max_episode_steps)

        current_discount_tensor = self.discount_tensor[:max_episode_len]

        # Cast sample batch to tensor.
        # Shape (num_episodes, env_max_episode_steps, obs_size)
        # We omit actions which do not appear in constraints of pollution environment.

        obs_size = train_batch_with_obs_and_actions['obs'].shape[1]
        action_size = train_batch_with_obs_and_actions['actions'].shape[1]
        num_episodes = len(train_batch_split_by_episode)
        obs_split_by_episode = torch.zeros((num_episodes, max_episode_len, obs_size)).cuda()
        # action_split_by_episode = torch.zeros((num_episodes, max_episode_len, action_size)).cuda()
        for episode_idx, sample_episode in enumerate(train_batch_split_by_episode):
            episode_len = sample_episode['obs'].shape[0]
            obs_split_by_episode[episode_idx, :episode_len, :] = torch.as_tensor(
                sample_episode['obs']
            ).cuda()
            # action_split_by_episode[episode_idx, :episode_len, :] = torch.as_tensor(
            #     sample_episode['actions']
            # ).cuda()
        # TODO: The code is only valid for pollution environment which does not require action.
        action_split_by_episode = torch.empty((num_episodes, max_episode_len, action_size)).cuda()

        dim_Y = self.env_local.dim_Y

        # TODO: You can save more time by implementing the following class more efficiently.
        # TODO: Also, you can use jax etc to compute gradient and hessian.
        class UMinusVcMonteCarlo:
            def __init__(inner_self):
                # TODO: Save intermediate resultes here to save time.
                return

            def func(inner_self, y: ndarray) -> float:
                y_tensor = torch.as_tensor(y).cuda()
                c_func_on_traj = self.env_local.c_func_on_traj(
                    batch_obs=obs_split_by_episode.view(num_episodes * max_episode_len, obs_size),
                    batch_action=action_split_by_episode.view(num_episodes * max_episode_len, action_size),
                    y=y_tensor
                ).reshape(num_episodes, max_episode_len)
                Vc_monte_carlo = torch.mean(
                    torch.sum(c_func_on_traj * current_discount_tensor[None, :], dim=1)
                ).item()
                u_y = self.env_local.u_func(y=y)
                return u_y - Vc_monte_carlo

            def grad(inner_self, y: ndarray) -> ndarray:
                y_tensor = torch.as_tensor(y).cuda()
                c_grad_on_traj = self.env_local.c_grad_on_traj(
                    batch_obs=obs_split_by_episode.view(num_episodes * max_episode_len, obs_size),
                    batch_action=action_split_by_episode.view(num_episodes * max_episode_len, action_size),
                    y=y_tensor
                ).reshape(num_episodes, max_episode_len, dim_Y)
                Vc_monte_carlo_grad = torch.mean(
                    torch.sum(c_grad_on_traj * current_discount_tensor[None, :, None], dim=1),
                    dim=0
                ).cpu().numpy()
                u_y_grad = self.env_local.u_grad(y=y)
                return u_y_grad - Vc_monte_carlo_grad

            def hess(inner_self, y: ndarray) -> ndarray:
                y_tensor = torch.as_tensor(y).cuda()
                c_hess_on_traj = self.env_local.c_hess_on_traj(
                    batch_obs=obs_split_by_episode.view(num_episodes * max_episode_len, obs_size),
                    batch_action=action_split_by_episode.view(num_episodes * max_episode_len, action_size),
                    y=y_tensor
                ).reshape(num_episodes, max_episode_len, dim_Y, dim_Y)
                Vc_monte_carlo_hess = torch.mean(
                    torch.sum(c_hess_on_traj * current_discount_tensor[None, :, None, None], dim=1),
                    dim=0
                ).cpu().numpy()
                u_y_hess = self.env_local.u_hess(y=y)
                return u_y_hess - Vc_monte_carlo_hess

        u_minus_Vc_monte_carlo_instance = UMinusVcMonteCarlo()
        opt_res = minimize(
            fun=u_minus_Vc_monte_carlo_instance.func,
            x0=self.init_y_for_optimize,
            jac=u_minus_Vc_monte_carlo_instance.grad,
            hess=u_minus_Vc_monte_carlo_instance.hess,
            bounds=self.env_local.bound_Y,
            method='trust-constr',
        )
        if not opt_res.success:
            print('Optimization result:')
            print(opt_res)
            raise OptimizeError('Fail to find the max violat y.')
        max_avg_constraint_value_violat_y = opt_res.x.copy()
        self.init_y_for_optimize = opt_res.x.copy()
        max_avg_constraint_value_violat = -opt_res.fun

        return max_avg_constraint_value_violat, max_avg_constraint_value_violat_y, all_success_flag

    def set_constraint_information_for_max_violat_y(self, policy_to_train: SICPPOPolicy,
                                                    sample_batch: MultiAgentBatch, y: ndarray) -> MultiAgentBatch:
        y_tensor = torch.as_tensor(y, dtype=torch.float).cuda()

        reduce_batch_on_device = {
            'obs': torch.as_tensor(sample_batch[DEFAULT_POLICY_IDX]['obs'], dtype=torch.float).cuda(),
            'actions': torch.as_tensor(sample_batch[DEFAULT_POLICY_IDX]['actions'], dtype=torch.float).cuda()
        }

        # Compute c_y.
        sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS] = \
            self.env_local.batch_c(batch_y=y_tensor[None, :],
                                   batch_obs=reduce_batch_on_device['obs'],
                                   batch_action=reduce_batch_on_device['actions']).cpu().numpy()
        # Compute value function of constraints.
        sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_VALUE] = \
            policy_to_train.compute_constraint_val_for_sample_batch(
                input_dict=reduce_batch_on_device,
                batch_y_tensor=y_tensor[None, :]
            )
        # Compute standardized advantage and target value (Q) of constraints.
        # The following code is computed only on cpu.
        sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_ADVANTAGE], \
            sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_VALUE_TARGET] = \
            policy_to_train.compute_standardized_gae_and_value_target_for_constraints(
                sample_batch=sample_batch[DEFAULT_POLICY_IDX],
                max_violat_flag=True
            )
        # Squeeze the result.
        sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS] = sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS].squeeze()
        sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_VALUE] = sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_VALUE].squeeze()
        sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_ADVANTAGE] = sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_ADVANTAGE].squeeze()
        sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_VALUE_TARGET] = sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_VALUE_TARGET].squeeze()
        return sample_batch
