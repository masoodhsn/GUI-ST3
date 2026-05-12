from stable_baselines3 import DQN
import gymnasium as gym

from learning_algorithm.base_algorithm import (
    BaseRLAlgorithm
)


class CustomDQN(
    DQN,
    BaseRLAlgorithm
):

    algorithm_name = "DQN"


    @classmethod
    def get_init_params(cls):

        return {
            "learning_rate": 0.0003,
            "gamma": 0.99
        }


    @classmethod
    def is_compatible(cls, env):

        return isinstance(
            env.action_space,
            gym.spaces.Discrete
        )