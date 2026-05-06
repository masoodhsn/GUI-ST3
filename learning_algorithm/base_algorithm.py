from abc import ABC
import numpy as np


class BaseRLAlgorithm(ABC):

    algorithm_name = "Base"

    @classmethod
    def is_compatible(cls, env):
        return True
    
    @classmethod
    def get_init_params(cls):
        """
        Returns metadata for UI + constructor
        """
        return {}
    

    default_policy = "MlpPolicy"

    @classmethod
    def from_params(cls, env, params):
        """
        Unified constructor for all algorithms
        """
        return cls(
            policy=cls.default_policy,
            env=env,
            **params
        )


    def __init__(self, *args, **kwargs):
        pass

    # -------------------
    # Environment helpers
    # -------------------
    def _reset_env(self):

        result = self.env.reset()

        # Gymnasium -> (obs, info)
        if isinstance(result, tuple):
            return result[0]

        # VecEnv -> np.ndarray
        if isinstance(result, np.ndarray):
            return result[0]

        return result

    def _step_env(self, action):

        # VecEnv
        if hasattr(self.env, "num_envs"):

            obs, reward, done, info = self.env.step([action])

            return (
                obs[0],
                reward[0],
                done[0]
            )

        # Gymnasium
        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        return (
            obs,
            reward,
            done
        )

    # -------------------
    # Policy helpers
    # -------------------
    def _select_action(self, state):

        action, _ = self.predict(
            state,
            deterministic=False
        )

        return action

    # -------------------
    # Main API
    # -------------------
    def learn(self, *args, **kwargs):

        if hasattr(super(), "learn"):
            return super().learn(
                *args,
                **kwargs
            )

        raise NotImplementedError

    def predict(self, *args, **kwargs):

        if hasattr(super(), "predict"):
            return super().predict(
                *args,
                **kwargs
            )

        raise NotImplementedError

    # -------------------
    # Save / Load
    # -------------------
    def save(self, *args, **kwargs):

        if hasattr(super(), "save"):
            return super().save(
                *args,
                **kwargs
            )

        raise NotImplementedError

    @classmethod
    def load(cls, *args, **kwargs):

        if hasattr(super(cls, cls), "load"):
            return super(cls, cls).load(
                *args,
                **kwargs
            )

        raise NotImplementedError