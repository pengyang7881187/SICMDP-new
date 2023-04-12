import numpy as np

UNIT_SPEED = 0.1

GOAL_EPS = UNIT_SPEED

STEP_COST_COEFF = 0.1

GOAL_REWARD = 5.

MAX_STEPS = 100

LR = 1e-2

ETA = 0.1

SAMPLE_Y_SIZE = 100

SEED = 74751

GAMMA = 1.

CHECK_FINENESS = 32

SEPARATION_LEN = 50

DEFAULT_POLLUTION_CONFIG = {
    'name': 'DefaultPollutionEnvironment',
    'seed': 74751,
    'dim_Y': 2,
    'lb_Y': np.zeros((2,)),
    'ub_Y': np.ones((2,)),
    'start_pos': np.zeros((2,)),
    'goal_pos': np.ones((2,)),
    'protect_poss': 0.5 * np.ones((1, 2)),
    'unit_speed': UNIT_SPEED,
    'goal_eps': GOAL_EPS,
    'max_steps': MAX_STEPS,
    'step_cost_coeff': STEP_COST_COEFF,
    'goal_reward': GOAL_REWARD,
    'pollution_func_name': 'exponential',
    'protect_func_name': 'exponential',
    'pollution_func_parameter': 20.,
    'protect_func_parameter': 20.,
    'protect_func_multiplier': 0.005,
    'protect_func_intercept': 0.02,
    'record_pos_flag': True,
    'constant_speed_flag': True,
    'symmetric_flag': True,
}

