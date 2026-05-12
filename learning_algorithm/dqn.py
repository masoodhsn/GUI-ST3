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
    def get_hyperparameters(cls):
        return {
            "learning_rate": {
                "default": 0.0003,
                "type": float
            },
            "gamma": {
                "default": 0.99,
                "type": float
            }
        }


    @classmethod
    def is_compatible(cls, env):

        return isinstance(
            env.action_space,
            gym.spaces.Discrete
        )