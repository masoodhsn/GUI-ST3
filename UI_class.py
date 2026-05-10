from custom_games.CartPole.env import CustomEnv
from learning_algorithm.dqn import CustomDQN

import time
from PIL import Image

from callback import StreamlitTrainingCallback


class UI():

    def __init__(self):

        print("UI initialized")

        self.env = CustomEnv()

        self.model = CustomDQN(
            "MlpPolicy",
            self.env,
            **CustomDQN.get_init_params(),
            verbose=1
        )

        self.last_frame = None


    def is_compatible(self):

        print(
            "Algorithm compatible:",
            self.model.is_compatible(
                self.env
            )
        )


    def run(self, display):

        obs, info = self.env.reset()

        episode_reward = 0

        for _ in range(1000):

            action, _ = self.model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                self.env.step(action)
            )

            episode_reward += reward

            frame = self.env.render()

            display.image(Image.fromarray(frame))

            time.sleep(0.03)   # 30 FPS

            if terminated or truncated:

                print(
                    "Episode reward:",
                    episode_reward
                )

                episode_reward = 0

                obs, info = self.env.reset()


    def train(self, chart):

        callback = StreamlitTrainingCallback(
            chart_placeholder=chart
        )

        self.model.learn(
            total_timesteps=70000,
            callback=callback
        )


    def save(self):

        self.model.save("test")


    def load(self):

        self.model = CustomDQN.load(
            "test",
            env=self.env
        )