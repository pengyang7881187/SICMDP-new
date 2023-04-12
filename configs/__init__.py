from . import pollution_env_config

YOUR_WORKING_DIR = '.'
SAVE_CHECKPOINT_DIR = f'{YOUR_WORKING_DIR}/saved_checkpoint'
PLOT_DIR = f'{YOUR_WORKING_DIR}/figures'
RAY_RESULTS_DIR = f'{YOUR_WORKING_DIR}/ray_results'
SAVE_TENSORBOARD_RESULT_DIR = f'{YOUR_WORKING_DIR}/ray_results_df'

from .pollution_env_config import DEFAULT_POLLUTION_CONFIG
