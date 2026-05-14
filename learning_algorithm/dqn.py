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
    

    def export_training_state(self):

        return {

            # neural network weights
            "policy_state":
                self.policy.state_dict(),

            # optimizer state
            "optimizer_state":
                self.policy.optimizer.state_dict(),

            # replay memory
            "replay_buffer":
                self.replay_buffer,

            # training progress
            "num_timesteps":
                self.num_timesteps,

            # epsilon-greedy progress
            "exploration_rate":
                self.exploration_rate
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

        self.replay_buffer = (
            state["replay_buffer"]
        )

        self.num_timesteps = (
            state["num_timesteps"]
        )

        self.exploration_rate = (
            state["exploration_rate"]
        )