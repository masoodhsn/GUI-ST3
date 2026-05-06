import gymnasium as gym

class CustomEnv(gym.Env):
    """
    Custom wrapper for FrozenLake-v1
    """

    def __init__(self):
        # is_slippery=False makes the environment deterministic for easier testing
        self.env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=True)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # metadata for algorithm selection
        self.state_space_type = 'discrete' if isinstance(self.observation_space, gym.spaces.Discrete) else 'continuous'
        self.action_space_type = 'discrete' if isinstance(self.action_space, gym.spaces.Discrete) else 'continuous'
        self.episode_length = 100  
        self.reward_type = 'sparse'  

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
