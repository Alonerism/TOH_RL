import logging
from gymnasium.envs.registration import register
# from gym_hanoi.envs.hanoi_env import HanoiGoalEnv

logger = logging.getLogger(__name__)

register(
    id='Hanoi-v0',
    entry_point='gym_hanoi.envs:HanoiEnv',
)
