from stable_baselines3 import PPO
import gymnasium as gym

from learning_algorithm.base_algorithm import (
    BaseRLAlgorithm
)


class CustomPPO(
    PPO,
    BaseRLAlgorithm
):

    algorithm_name = "PPO"


    @classmethod
    def get_init_params(cls):

        return {
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "n_steps": 2048
        }


    @classmethod
    def is_compatible(cls, env):

        return True