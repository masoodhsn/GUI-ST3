from custom_games.CartPole.env import CustomEnv

import time
from PIL import Image

from callback import StreamlitTrainingCallback

from pathlib import Path
import importlib


class Backend:

    def __init__(self, algorithm):


        module = importlib.import_module(
            f"learning_algorithm.{algorithm}"
        )

        self.algorithm_class = getattr(
            module,
            f"Custom{algorithm.upper()}"
        )

        self.hyperparams = self.algorithm_class.get_hyperparameters()

        self.env = CustomEnv()

        self.model = self.build_model()

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

        self.algorithm_class = getattr(
            module,
            f"Custom{algorithm_name.upper()}"
        )

        self.hyperparams = self.algorithm_class.get_hyperparameters()

        return self.algorithm_class



    def update_hyperparams(self, new_params):

        for k, v in new_params.items():
            if k in self.hyperparams:
                self.hyperparams[k]["default"] = v



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
        total_timesteps
    ):

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


    def load(self, uploaded_file):

        if uploaded_file is None:
            return

        self.model = self.algorithm_class.load(
            uploaded_file,
            env=self.env
        )


    def rebuild_model(self):

        old_state = (
            self.model.export_training_state()
        )

        self.build_model()

        self.model.import_training_state(
            old_state
        )
    

    def build_model(self):

        params = {
            k: v["default"]
            for k, v in self.hyperparams.items()
        }

        self.model = self.algorithm_class(
            "MlpPolicy",
            self.env,
            **params,
            verbose=1
        ) 

        return self.model