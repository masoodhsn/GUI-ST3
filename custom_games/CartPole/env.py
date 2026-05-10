import gymnasium as gym


class CustomEnv(gym.Env):

    def __init__(self):

        self.env = gym.make(
            "CartPole-v1",
            render_mode="rgb_array"
        )

        # spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # metadata for your platform
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

        # cartpole default max steps
        self.episode_length = 500

        # reward every correct step = dense
        self.reward_type = "dense"

    def reset(self, **kwargs):

        obs, info = self.env.reset(
            **kwargs
        )

        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )

    def render(self):
        frame = self.env.render()
        return frame.copy()

    def close(self):
        self.env.close()