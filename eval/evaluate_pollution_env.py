import warnings
import numpy as np
import matplotlib.pyplot as plt

from envs import SICMDPEnv
from utils import DEFAULT_POLICY_IDX
from algorithms.sicppo import SICPPO, OptimizeError
from configs.pollution_env_config import MAX_STEPS

from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import concat_samples
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes


def get_batch_mirror_pos(batch_pos):
    assert batch_pos.ndim == 2 and batch_pos.shape[1] == 2
    e = np.ones((2,)) / np.sqrt(2)
    batch_size = batch_pos.shape[0]
    batch_project_pos = np.inner(e, batch_pos)[:, None] * np.repeat(e[None, :], repeats=batch_size, axis=0)
    batch_mirror_pos = 2 * batch_project_pos - batch_pos
    return batch_mirror_pos


def eval_pollution_env_function(algorithm: SICPPO, eval_workers: WorkerSet):
    """Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    """

    optimize_y_flag = algorithm.config['evaluation_config']['eval_optimize_y_flag']
    evaluation_episodes = algorithm.config['evaluation_duration']
    num_workers = algorithm.config['evaluation_num_workers']
    render_flag = algorithm.config['evaluation_config']['render_flag']
    assert algorithm.config['evaluation_duration_unit'] == 'episodes'

    local_worker = eval_workers.local_worker()

    if num_workers == 0:
        sample_batch = concat_samples([local_worker.sample() for _ in range(evaluation_episodes)])
    else:
        assert evaluation_episodes % num_workers == 0
        iter_num = evaluation_episodes // num_workers
        sample_batch_lst = []
        for _ in range(iter_num):
            sample_batch_lst.extend(eval_workers.foreach_worker(
                lambda w: w.sample(), local_worker=False, healthy_only=True
            ) for _ in range(iter_num))
        sample_batch = concat_samples(sample_batch_lst)

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes = collect_episodes(workers=eval_workers, timeout_seconds=99999)
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(episodes)

    # TODO: Action is lost here, try to recover it, we use a placeholder here, since it is not used later.
    sample_batch[DEFAULT_POLICY_IDX]['actions'] = np.empty_like(sample_batch[DEFAULT_POLICY_IDX]['obs'])

    if optimize_y_flag:
        try:
            max_violat_val, max_violat_y, _ = \
                algorithm.get_optimal_max_avg_monte_carlo_constraint_violat_value_and_pos_and_all_success_flag(
                    train_batch=sample_batch[DEFAULT_POLICY_IDX]
                )
        except OptimizeError as optimize_error:
            warnings.warn(f'Warning: Fail to optimize y with error information: {optimize_error}, use grid search instead.')
            grid_batch_y = algorithm.env_local.grid.generate(fineness=10)
            sample_batch[DEFAULT_POLICY_IDX][SICMDPEnv.CONSTRAINTS] = \
                algorithm.env_local.batch_c(batch_y=grid_batch_y,
                                            batch_obs=sample_batch[DEFAULT_POLICY_IDX]['obs'],
                                            batch_action=sample_batch[DEFAULT_POLICY_IDX]['actions'])
            max_violat_val, max_violat_idx, _ = \
                algorithm.get_sample_max_avg_monte_carlo_constraint_violat_value_and_idx_and_all_success_flag(
                    train_batch=sample_batch[DEFAULT_POLICY_IDX], batch_y=grid_batch_y
                )
            max_violat_y = grid_batch_y[max_violat_idx]
        metrics['optimize_max_violat_val'] = max_violat_val
        metrics['optimize_max_violat_y'] = max_violat_y
    else:
        raise NotImplementedError

    sample_batch_split_by_episode = sample_batch[DEFAULT_POLICY_IDX].split_by_episode()
    episode_len_lst = [sample_episode.agent_steps() for sample_episode in sample_batch_split_by_episode]
    episode_len_min = min(episode_len_lst)
    episode_len_max = max(episode_len_lst)
    metrics['episode_len_min'] = episode_len_min
    metrics['episode_len_max'] = episode_len_max

    # Only work for pollution env.
    if render_flag:
        episode_len_min_idx = np.argmin(episode_len_lst)
        discrete_path = sample_batch_split_by_episode[episode_len_min_idx]['obs'][:, :2]
        if len(discrete_path) < MAX_STEPS:
            discrete_path = np.concatenate([discrete_path, np.array([[1., 1.], ])])
        discrete_predefined_path = algorithm.config['evaluation_config']['predefined_path']
        plt.figure()
        plt.plot(discrete_path[:, 0], discrete_path[:, 1], marker='o', mec='none', ms=4, lw=1, label='Policy')
        plt.plot(discrete_predefined_path[:, 0], discrete_predefined_path[:, 1], marker='o', mec='none', ms=4, lw=1,
                 label='Predefined')
        mirror_discrete_predefined_path = get_batch_mirror_pos(discrete_predefined_path)
        plt.plot(mirror_discrete_predefined_path[:, 0], mirror_discrete_predefined_path[:, 1], marker='o', mec='none', ms=4, lw=1,
                 label='Predefined')
        plt.legend()
        plt.show()
    return metrics