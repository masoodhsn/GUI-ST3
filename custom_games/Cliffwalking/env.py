import gymnasium as gym

class CustomEnv(gym.Env):
    """
    Custom wrapper for CliffWalking-v1 environment for testing.
    Compatible with RL platform.
    """

    def __init__(self):
        # Create the original CliffWalking environment
        self.env = gym.make("CliffWalking-v1", render_mode="rgb_array")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    # -------------------------------
    # Reset
    # -------------------------------
    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    # -------------------------------
    # Step
    # -------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    # -------------------------------
    # Render
    # -------------------------------
    def render(self):
        return self.env.render()

    # -------------------------------
    # Close
    # -------------------------------
    def close(self):
        self.env.close()