import logging
import numpy as np
from torch import Tensor
from numpy import ndarray
from typing import Dict, List, Type, Union, Tuple

from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)

from envs import SICMDPEnv
from utils import ArrayType
from models import SICPPOFCNet
from .process import batch_discount_cumsum, batch_z_score_normalization

import torch

logger = logging.getLogger(__name__)


class SICPPOPolicy(
    ValueNetworkMixin,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    KLCoeffMixin,
    TorchPolicyV2,
):
    """PyTorch policy class used with SICPPO."""

    def __init__(self, observation_space, action_space, config):
        validate_config(config)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        self._initialize_loss_from_dummy_batch()

        # New attributes for SICPPOPolicy.
        self.dim_Y = config['env_config']['dim_Y']
        # You have to specify them in every training loop.
        self.current_optimize_y_flag = config['optimize_y_flag']
        self.all_success_flag = False
        self.max_violat_val = np.inf
        self.optimize_max_violat_val = np.inf
        self.max_violat_y = np.empty((self.dim_Y,))
        self.batch_y = np.empty((1, self.dim_Y))

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[Tensor, List[Tensor]]:
        """Compute loss for Semi-infinite Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """
        if SICMDPEnv.CONSTRAINTS not in train_batch:
            return self.vanilla_ppo_loss(model, dist_class, train_batch)
        model: SICPPOFCNet

        update_val_flag = True
        # if (not all_fail_flag) and max_violat_val > self.config['eta']:
        if self.all_success_flag and self.max_violat_val > self.config['eta']:
            # Constraint violation! Update constraint instead of reward.
            update_val_flag = False

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            raise NotImplementedError('RNN is not supported!')
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        # batch_size, num_y = train_batch[SICMDPEnv.CONSTRAINTS_VALUE].shape

        # Negative constraint advantage of max violat y.
        # Shape (batch_size)
        if not self.current_optimize_y_flag:
            negative_max_violat_constraint_advantage = \
                -train_batch[SICMDPEnv.CONSTRAINTS_ADVANTAGE][:, self.max_violat_idx]
        else:
            negative_max_violat_constraint_advantage = \
                -train_batch[SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_ADVANTAGE]

        if update_val_flag:
            # Update reward.
            surrogate_loss = torch.min(
                train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
                train_batch[Postprocessing.ADVANTAGES]
                * torch.clamp(
                    logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
                ),
            )
        else:
            # Update constraints.
            surrogate_loss = torch.min(
                negative_max_violat_constraint_advantage
                * logp_ratio,
                negative_max_violat_constraint_advantage
                * torch.clamp(
                    logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
                ),
            )

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            # Placeholder.
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        # Compute a constraint value function loss.
        if self.config["use_critic"]:
            # Shape (batch_size, num_y+1)
            constraint_value_fn_out = model.constraint_value_function(
                batch_y_tensor=torch.as_tensor(np.concatenate([self.batch_y, self.max_violat_y[None, :]], axis=0),
                                               device=surrogate_loss.device, dtype=torch.float),
                input_dict=train_batch
            )
            # Shape (batch_size)
            # Max violat constraint.
            max_violat_constraint_value_fn_out = constraint_value_fn_out[:, -1]
            max_violat_constraint_vf_loss = torch.pow(
                max_violat_constraint_value_fn_out - train_batch[SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_VALUE_TARGET], 2.0
            )
            max_violat_constraint_vf_loss_clipped = torch.clamp(max_violat_constraint_vf_loss, 0, self.config["constraint_vf_clip_param"])
            mean_max_violat_constraint_vf_loss = reduce_mean_valid(max_violat_constraint_vf_loss_clipped)

            # Shape (batch_size, num_y)
            # Other constraints.
            constraint_value_fn_out = constraint_value_fn_out[:, :-1]
            constraint_vf_loss = torch.pow(
                constraint_value_fn_out - train_batch[SICMDPEnv.CONSTRAINTS_VALUE_TARGET], 2.0
            )
            constraint_vf_loss_clipped = torch.clamp(constraint_vf_loss, 0, self.config["constraint_vf_clip_param"])
            mean_constraint_vf_loss = reduce_mean_valid(constraint_vf_loss_clipped)
        # Ignore the value function.
        else:
            constraint_value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            # Placeholder.
            constraint_vf_loss_clipped = mean_constraint_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = self.config["policy_loss_coeff"] * reduce_mean_valid(-surrogate_loss) + self.config["vf_loss_coeff"] * mean_vf_loss + \
                     self.config["constraint_vf_loss_coeff"] * mean_constraint_vf_loss - \
                     self.entropy_coeff * mean_entropy + self.config["max_constraint_vf_loss_coeff"] * mean_max_violat_constraint_vf_loss
        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["mean_constraint_vf_loss"] = mean_constraint_vf_loss
        model.tower_stats["max_violat_mean_constraint_vf_loss"] = mean_max_violat_constraint_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["max_violat_constraint_vf_explained_var"] = explained_variance(
            train_batch[SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_VALUE_TARGET], max_violat_constraint_value_fn_out
        )
        model.tower_stats["constraint_vf_explained_var"] = explained_variance(
            train_batch[SICMDPEnv.CONSTRAINTS_VALUE_TARGET].reshape(-1), constraint_value_fn_out.reshape(-1)
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        model.tower_stats["max_constraint_violat"] = torch.tensor(self.max_violat_val).to(total_loss.device)
        model.tower_stats["optimize_max_violat_val"] = torch.tensor(self.optimize_max_violat_val).to(total_loss.device)
        return total_loss

    def vanilla_ppo_loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[Tensor, List[Tensor]]:
        """Compute loss for Semi-infinite Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            raise NotImplementedError('RNN is not supported!')
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        # Set a default value.
        model.tower_stats["mean_constraint_vf_loss"] = torch.tensor(np.nan).to(total_loss.device)
        model.tower_stats["constraint_vf_explained_var"] = torch.tensor(np.nan).to(total_loss.device)
        model.tower_stats["max_constraint_violat"] = torch.tensor(np.nan).to(total_loss.device)
        model.tower_stats["max_violat_mean_constraint_vf_loss"] = torch.tensor(np.nan).to(total_loss.device)
        model.tower_stats["max_violat_constraint_vf_explained_var"] = torch.tensor(np.nan).to(total_loss.device)
        model.tower_stats["optimize_max_violat_val"] = torch.tensor(np.nan).to(total_loss.device)
        return total_loss

    @override(TorchPolicyV2)
    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, ArrayType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "train_max_constraint_violat": torch.mean(
                    torch.stack(self.get_tower_stats("max_constraint_violat"))
                ),
                "optimize_max_violat_val": torch.mean(
                    torch.stack(self.get_tower_stats("optimize_max_violat_val"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "constraint_vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_constraint_vf_loss"))
                ),
                "constraint_vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("constraint_vf_explained_var"))
                ),
                "train_max_violat_mean_constraint_vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("max_violat_mean_constraint_vf_loss"))
                ),
                "train_max_violat_constraint_vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("max_violat_constraint_vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy_coeff": self.entropy_coeff,
            }
        )

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        with torch.no_grad():
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )

    # Call constraint_value_function with no gradient and return ndarray.
    def compute_constraint_val_for_sample_batch(self, input_dict: Dict[str, Tensor], batch_y_tensor: Tensor) -> ndarray:
        with torch.no_grad():
            constraint_val = self.model.constraint_value_function(batch_y_tensor=batch_y_tensor,
                                                                  input_dict=input_dict).cpu().numpy()
            return constraint_val

    # To call this method, you should compute constraint and constraint value in advance.
    def compute_standardized_gae_and_value_target_for_constraints(
            self,
            sample_batch: SampleBatch,
            max_violat_flag=False
    ) -> Tuple[ndarray, ndarray]:

        # The method is modified from compute_gae_for_sample_batch and compute_advantage.
        CONSTRAINTS = SICMDPEnv.CONSTRAINTS
        CONSTRAINTS_VALUE = SICMDPEnv.CONSTRAINTS_VALUE
        if max_violat_flag:
            CONSTRAINTS = SICMDPEnv.MAX_VIOLAT_CONSTRAINTS
            CONSTRAINTS_VALUE = SICMDPEnv.MAX_VIOLAT_CONSTRAINTS_VALUE

        # Trajectory is actually complete -> last c=0.0.
        if sample_batch[SampleBatch.TERMINATEDS][-1]:
            last_c = np.zeros_like(sample_batch[CONSTRAINTS_VALUE][0])
        # Trajectory has been truncated -> last c=VF estimate of last obs.
        else:
            if self.config["_enable_rl_module_api"]:
                raise NotImplementedError('_enable_rl_trainer_api is not supported!')
            else:
                last_c = sample_batch[CONSTRAINTS_VALUE][-1]

        gamma = self.config["gamma"]
        lambda_ = self.config["lambda"]
        use_gae = self.config["use_gae"],
        use_critic = self.config.get("use_critic", True)

        assert (
                CONSTRAINTS_VALUE in sample_batch or not use_critic
        ), "use_critic=True but constraint values not found"
        assert use_critic or not use_gae, "Can't use gae without using a constraint critic."

        if use_gae:
            c_vpred_t = np.concatenate([sample_batch[CONSTRAINTS_VALUE], last_c.reshape(1, -1)], axis=0)
            delta_t = sample_batch[CONSTRAINTS] + gamma * c_vpred_t[1:] - c_vpred_t[:-1]
            # This formula for the advantage comes from:
            # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
            constraints_advantages = batch_discount_cumsum(delta_t, gamma * lambda_)
            # Q function for constraints.
            constraints_value_target = (
                    constraints_advantages + sample_batch[CONSTRAINTS_VALUE]
            ).astype(np.float32)
        else:
            constraints_plus_c_v = np.concatenate(
                [sample_batch[CONSTRAINTS], last_c.reshape(1, -1)]
            )
            # Monte-Carlo estimate.
            discounted_returns = batch_discount_cumsum(constraints_plus_c_v, gamma)[:-1].astype(
                np.float32
            )

            if use_critic:
                # Traditional advantage: Monte-Carlo Q and critic V.
                constraints_advantages = (
                        discounted_returns - sample_batch[CONSTRAINTS_VALUE]
                )
                constraints_value_target = discounted_returns
            else:
                # No critic (hence baseline) is provided, substitute advantage with Q.
                constraints_advantages = discounted_returns
                constraints_value_target = np.zeros_like(
                    constraints_advantages
                )
        constraints_advantages = batch_z_score_normalization(
            constraints_advantages
        ).astype(np.float32)
        return constraints_advantages, constraints_value_target
