from stable_baselines3.common.callbacks import (
    BaseCallback
)

import os


class StreamlitTrainingCallback(
    BaseCallback
):

    def __init__(
        self,
        chart_placeholder,
        save_path="saved_models",
        verbose=0
    ):

        super().__init__(verbose)

        self.chart_placeholder = (
            chart_placeholder
        )

        self.save_path = save_path

        self.rewards = []

        self.chart_data = []

        self.current_reward = 0

        self.best_reward = float("-inf")


        os.makedirs(
            self.save_path,
            exist_ok=True
        )


    def _on_step(self):

        reward = self.locals[
            "rewards"
        ][0]

        done = self.locals[
            "dones"
        ][0]

        self.current_reward += reward


        if done:

            self.rewards.append(
                self.current_reward
            )

            self.current_reward = 0


            # update every 20 episod
            if len(self.rewards) % 20 == 0:

                # moving average
                window = self.rewards[-20:]

                avg_reward = (
                    sum(window)
                    / len(window)
                )

                self.chart_data.append(
                    avg_reward
                )

                self.chart_placeholder.line_chart(
                    self.chart_data
                )

                print(
                    "Average Reward:",
                    avg_reward
                )


                # save best model
                if avg_reward > self.best_reward:

                    self.best_reward = (
                        avg_reward
                    )

                    save_name = os.path.join(
                        self.save_path,
                        "best_model"
                    )

                    self.model.save(
                        save_name
                    )

                    print(
                        f"New best model saved! "
                        f"Reward={avg_reward}"
                    )


        return True