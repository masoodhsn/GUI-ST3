from stable_baselines3 import PPO

from learning_algorithm.base_algorithm import BaseRLAlgorithm


class CustomPPO(PPO, BaseRLAlgorithm):

    algorithm_name = "PPO"

    @classmethod
    def is_compatible(cls, env):

        return True
    
    @classmethod
    def get_init_params(cls):
        return {
            "learning_rate": 0.0003,
            "gamma": 0.99
        }