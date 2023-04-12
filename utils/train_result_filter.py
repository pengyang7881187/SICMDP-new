from typing import Dict

RESULT_FILTER = (
    'training_iteration',
    'episode_len_mean',
    'episode_reward_max',
    'episode_reward_mean',
    'episode_reward_min',
    'episodes_this_iter',
    'episodes_total',
    'time_this_iter_s',
    'time_total_s',
)

EVALUATION_FILTER = (
    'episode_len_max',
    'episode_len_mean',
    'episode_len_min',
    'episode_reward_max',
    'episode_reward_mean',
    'episode_reward_min',
    'episodes_this_iter',
    'optimize_max_violat_val',
    'optimize_max_violat_y',
)

INNER_RESULT_FILTER = (
    'cur_lr',
    'optimize_max_violat_val',
    'train_max_constraint_violat',
    'total_loss',
    'policy_loss',
    'vf_loss',
    'constraint_vf_loss',
    'train_max_violat_mean_constraint_vf_loss',
    'vf_explained_var',
    'constraint_vf_explained_var',
    'train_max_violat_constraint_vf_explained_var',
    'kl',
)

def filter_result(result: Dict):
    filtered_dict = {key: result[key] for key in RESULT_FILTER}
    inner_result = result['info']['learner']['default_policy']['learner_stats']
    filtered_dict.update({key: inner_result[key] for key in INNER_RESULT_FILTER})
    if 'evaluation' in result:
        evaluation_dict = result['evaluation']
        filtered_evaluation_dict = {key: evaluation_dict[key] for key in EVALUATION_FILTER}
        filtered_dict.update({'evaluation': filtered_evaluation_dict})
    return filtered_dict