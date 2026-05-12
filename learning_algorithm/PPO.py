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
    def get_hyperparameters(cls):
        return {
            "learning_rate": {
                "default": 0.0003,
                "type": float
            },
            "gamma": {
                "default": 0.99,
                "type": float
            },
            "n_steps": {
                "default": 2048,
                "type": int
            }
        }


    @classmethod
    def is_compatible(cls, env):

        return True