import gymnasium as gym
from custom_games.base_env import BaseEnv


class CustomEnv(BaseEnv):

    environment_name = "LunarLander"
    gym_env_name = "LunarLander-v2"

    def __init__(self):

        super().__init__()

        self.episode_length = 1000
        self.reward_type = "dense"