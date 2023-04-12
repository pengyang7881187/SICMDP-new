import torch
import logging
from typing import List, Optional, Union

from numpy import ndarray
from ray.rllib import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import (
    concat_samples,
)
from ray.rllib.utils.typing import SampleBatchType

from envs import SICMDPEnv
from utils import DEFAULT_POLICY_IDX
from .sicppo_policy import SICPPOPolicy

logger = logging.getLogger(__name__)


class SICMDPRolloutWorker(RolloutWorker):
    def sample_and_process_constraints(self, batch_y: ndarray) -> SampleBatchType:
        policy: SICPPOPolicy = self.get_policy(DEFAULT_POLICY_IDX)
        device = policy.device

        batch = self.sample()
        # batch_y shape (num_y, dim_Y)
        batch_y_tensor = torch.as_tensor(batch_y, device=device, dtype=torch.float)

        reduce_batch_on_device = {
            'obs': torch.as_tensor(batch[DEFAULT_POLICY_IDX]['obs'], device=device, dtype=torch.float),
            'actions': torch.as_tensor(batch[DEFAULT_POLICY_IDX]['actions'], device=device, dtype=torch.float)
        }

        # Compute batch_c_y.
        self.env: SICMDPEnv
        batch[DEFAULT_POLICY_IDX][SICMDPEnv.CONSTRAINTS] = \
            self.env.batch_c(batch_y=batch_y_tensor,
                             batch_obs=reduce_batch_on_device['obs'],
                             batch_action=reduce_batch_on_device['actions']).cpu().numpy()
        # Compute value function of constraints.
        batch[DEFAULT_POLICY_IDX][SICMDPEnv.CONSTRAINTS_VALUE] = policy.compute_constraint_val_for_sample_batch(
            input_dict=reduce_batch_on_device,
            batch_y_tensor=batch_y_tensor
        )
        # Compute standardized advantage and target value (Q) of constraints.
        # The following code is computed only on cpu.
        batch[DEFAULT_POLICY_IDX][SICMDPEnv.CONSTRAINTS_ADVANTAGE], \
            batch[DEFAULT_POLICY_IDX][SICMDPEnv.CONSTRAINTS_VALUE_TARGET] = \
            policy.compute_standardized_gae_and_value_target_for_constraints(
            sample_batch=batch[DEFAULT_POLICY_IDX],
        )
        return batch

def synchronous_parallel_sample_and_process_constraints(
    worker_set: WorkerSet,
    batch_y: ndarray,
    max_agent_steps: Optional[int] = None,
    max_env_steps: Optional[int] = None,
    concat: bool = True,
) -> Union[List[SampleBatchType], SampleBatchType]:
    """Runs parallel and synchronous rollouts on all remote workers.

    Waits for all workers to return from the remote calls.

    If no remote workers exist (num_workers == 0), use the local worker
    for sampling.

    Alternatively to calling `worker.sample.remote()`.

    Args:
        worker_set: The WorkerSet to use for sampling.
        batch_y: ndarray of shape (num_y, dim_Y)
            Batch of y.
        max_agent_steps: Optional number of agent steps to be included in the
            final batch.
        max_env_steps: Optional number of environment steps to be included in the
            final batch.
        concat: Whether to concat all resulting batches at the end and return the
            concat'd batch.

    Returns:
        The list of collected sample batch types (one for each parallel
        rollout worker in the given `worker_set`).

    """
    # Only allow one of `max_agent_steps` or `max_env_steps` to be defined.
    assert not (max_agent_steps is not None and max_env_steps is not None)

    agent_or_env_steps = 0
    max_agent_or_env_steps = max_agent_steps or max_env_steps or None
    all_sample_batches = []

    # Stop collecting batches as soon as one criterion is met.
    while (max_agent_or_env_steps is None and agent_or_env_steps == 0) or (
        max_agent_or_env_steps is not None
        and agent_or_env_steps < max_agent_or_env_steps
    ):
        # No remote workers in the set -> Use local worker for collecting
        # samples.
        if worker_set.num_remote_workers() <= 0:
            sample_batches = [SICMDPRolloutWorker.sample_and_process_constraints(worker_set.local_worker(), batch_y), ]
        # Loop over remote workers' `sample()` method in parallel.
        else:
            sample_batches = worker_set.foreach_worker(
                lambda w: w.sample_and_process_constraints(batch_y), local_worker=False, healthy_only=True
            )

            if worker_set.num_healthy_remote_workers() <= 0:
                # There is no point staying in this loop, since we will not be able to
                # get any new samples if we don't have any healthy remote workers left.
                break
        # Update our counters for the stopping criterion of the while loop.
        for b in sample_batches:
            if max_agent_steps:
                agent_or_env_steps += b.agent_steps()
            else:
                agent_or_env_steps += b.env_steps()
        all_sample_batches.extend(sample_batches)

    if concat is True:
        full_batch = concat_samples(all_sample_batches)
        # Discard collected incomplete episodes in episode mode.
        # if max_episodes is not None and episodes >= max_episodes:
        #    last_complete_ep_idx = len(full_batch) - full_batch[
        #        SampleBatch.DONES
        #    ].reverse().index(1)
        #    full_batch = full_batch.slice(0, last_complete_ep_idx)
        return full_batch
    else:
        return all_sample_batches