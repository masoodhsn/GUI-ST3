from stable_baselines3.common.callbacks import BaseCallback
from PIL import Image


class StreamlitTrainingCallback(BaseCallback):

    def __init__(
        self,
        render_env,
        display_placeholder,
        chart_placeholder,
        render_every=500,
        reward_window=100
    ):

        super().__init__()

        self.render_env = render_env
        self.display = display_placeholder
        self.chart = chart_placeholder

        self.render_every = render_every
        self.reward_window = reward_window

        self.episode_rewards = []
        self.current_obs = self.render_env.reset()[0]

    # ---------------------------
    # main hook
    # ---------------------------
    def _on_step(self):

        infos = self.locals.get("infos", [])

        # -----------------------
        # collect episode reward
        # -----------------------
        for info in infos:

            if "episode" in info:

                r = info["episode"]["r"]

                self.episode_rewards.append(r)

                # update chart every N episodes
                if len(self.episode_rewards) % self.reward_window == 0:

                    window = self.episode_rewards[-self.reward_window:]
                    mean_reward = sum(window) / len(window)

                    self.chart.line_chart(
                        self.episode_rewards
                    )

        # -----------------------
        # live render
        # -----------------------
        if self.n_calls % self.render_every == 0:

            action, _ = self.model.predict(
                self.current_obs,
                deterministic=True
            )

            action = int(action)

            self.current_obs, _, term, trunc, _ = self.render_env.step(action)

            if term or trunc:
                self.current_obs = self.render_env.reset()[0]

            frame = self.render_env.render()

            self.display.image(frame)

        return True