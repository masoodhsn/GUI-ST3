from custom_games.CartPole.env import CustomEnv
from learning_algorithm.dqn import CustomDQN

import time
from PIL import Image

from callback import StreamlitTrainingCallback

from pathlib import Path
import importlib


class Backend:

    def __init__(self, algorithm, learning_rate, gamma):

        self.env = CustomEnv()

        self.algorithm_class = self.set_algorithm(algorithm)

        self.current_params = (                    ###### hyper-parameters
            self.algorithm_class.get_init_params()
        )

        self.model = self.algorithm_class(
            "MlpPolicy",
            self.env,
            learning_rate= learning_rate,
            gamma= gamma,
            verbose=1
        )      

        print("Backend initialized")



    @staticmethod
    def get_available_algorithms():

        algo_path = Path(
            "learning_algorithm"
        )

        algorithms = []

        for file in algo_path.glob("*.py"):

            if file.stem in [
                "__init__",
                "base_algorithm"
            ]:
                continue

            algorithms.append(
                file.stem
            )

        return algorithms
    

    def set_algorithm(
        self,
        algorithm_name
    ):

        module = importlib.import_module(
            f"learning_algorithm.{algorithm_name}"
        )

        algorithm_class = getattr(
            module,
            f"Custom{algorithm_name.upper()}"
        )

        return algorithm_class





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

            time.sleep(0.05)  # 20 fps

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

        # self.model = self.algorithm_class.load(
        #     uploaded_file,
        #     env=self.env
        # )

        self.model = CustomDQN.load(
            uploaded_file,
            env=self.env
        )


    def rebuild_model(self):     

        old_model = self.model

        self.reset_model()

        self.model.policy.load_state_dict(
            old_model.policy.state_dict()
        )

        self.model.replay_buffer = (
            old_model.replay_buffer
        )
    

    def reset_model(self):
        
        self.model = self.algorithm_class(
            "MlpPolicy",
            self.env,
            **self.current_params,
            verbose=1
        )   
