from abc import ABC
import gymnasium as gym


class BaseEnv(gym.Env, ABC):

    environment_name = "BaseEnvironment"
    gym_env_name = None

    def __init__(self):

        self.env = self.build_env()

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.state_space_type = (
            "discrete"
            if isinstance(self.observation_space, gym.spaces.Discrete)
            else "continuous"
        )

        self.action_space_type = (
            "discrete"
            if isinstance(self.action_space, gym.spaces.Discrete)
            else "continuous"
        )

        self.episode_length = None
        self.reward_type = None


    def build_env(self):

        if self.gym_env_name is None:
            raise NotImplementedError(
                "gym_env_name or build_env() required"
            )

        return gym.make(
            self.gym_env_name,
            render_mode="rgb_array"
        )


    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


    def step(self, action):
        return self.env.step(action)


    def render(self):
        return self.env.render()


    def close(self):
        return self.env.close()