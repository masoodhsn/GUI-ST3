from custom_games.CartPole.env import CustomEnv
from learning_algorithm.dqn import CustomDQN

import time
from PIL import Image

from callback import StreamlitTrainingCallback


class Backend:

    def __init__(self, learning_rate, gamma):

        print("Backend initialized")

        self.env = CustomEnv()

        self.current_params = (                    ###### hyper-parameters
            CustomDQN.get_init_params()
        )

        self.model = CustomDQN(
            "MlpPolicy",
            self.env,
            learning_rate= learning_rate,
            gamma= gamma,
            verbose=1
        )
        

    def is_compatible(self):

        return self.model.is_compatible(
            self.env
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

            display.image(
                Image.fromarray(frame)
            )

            time.sleep(0.04)  # 25 fps

            if terminated or truncated:

                print(
                    "Episode reward:",
                    episode_reward
                )

                episode_reward = 0

                obs, info = self.env.reset()


    def train(
        self,
        chart,
        total_timesteps,
        learning_rate,
        gamma
    ):

        # update params
        self.current_params.update({
            "learning_rate": learning_rate,
            "gamma": gamma
        })

        # rebuild with new params
        self.rebuild_model()

        callback = StreamlitTrainingCallback(
            chart_placeholder=chart
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False
        )

        self.model.save(
            "test"
        )


    def load(self, uploaded_file):          ###### things need to be save to countinue learning

        if uploaded_file is None:
            return

        self.model = CustomDQN.load(
            uploaded_file,
            env=self.env
        )


    def rebuild_model(self):     

        old_model = self.model

        self.model = CustomDQN(
            "MlpPolicy",
            self.env,
            **self.current_params,
            verbose=1
        )

        self.model.policy.load_state_dict(
            old_model.policy.state_dict()
        )

        self.model.replay_buffer = (
            old_model.replay_buffer
        )
    

    def reset_model(self):

        self.model = CustomDQN(
            "MlpPolicy",
            self.env,
            **self.current_params,
            verbose=1
        )