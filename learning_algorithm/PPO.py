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
    

    def export_training_state(self):

         return {

            # network weights
            "policy_state":
                self.policy.state_dict(),

            # optimizer state
            "optimizer_state":
                self.policy.optimizer.state_dict(),

            # training progress
            "num_timesteps":
                self.num_timesteps
        }


    def import_training_state(
        self,
        state
    ):

        self.policy.load_state_dict(
            state["policy_state"]
        )

        self.policy.optimizer.load_state_dict(
            state["optimizer_state"]
        )

        self.num_timesteps = (
            state["num_timesteps"]
        )