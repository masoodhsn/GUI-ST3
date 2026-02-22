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

    def close(self):
        self.env.close()