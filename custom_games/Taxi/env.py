import gymnasium as gym

class CustomEnv(gym.Env):
    """
    Custom wrapper for Taxi-v3
    """

    def __init__(self):
        self.env = gym.make("Taxi-v3", render_mode="rgb_array")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()
import gymnasium as gym


class CustomEnv(gym.Env):

    def __init__(self):

        self.env = gym.make(
            "Taxi-v3",
            render_mode="rgb_array"
        )

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # metadata
        self.state_space_type = (
            "discrete"
            if isinstance(
                self.observation_space,
                gym.spaces.Discrete
            )
            else "continuous"
        )

        self.action_space_type = (
            "discrete"
            if isinstance(
                self.action_space,
                gym.spaces.Discrete
            )
            else "continuous"
        )

        self.episode_length = 200

        # rewards every step
        self.reward_type = "dense"

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
    def close(self):
        self.env.close()