import numpy as np
import pickle
import gymnasium as gym


class SARSA:
    def __init__(
        self,
        policy,
        env,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        verbose=0,
    ):
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        assert isinstance(env.action_space, gym.spaces.Discrete)

        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.verbose = verbose

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        # Q-table
        self.q_table = np.zeros((self.n_states, self.n_actions))

    # ----------------------------------------
    # ε-greedy policy
    # ----------------------------------------
    def _select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    # ----------------------------------------
    # Training
    # ----------------------------------------
    def learn(self, total_timesteps, report_every=100):
        """
        آموزش SARSA و yield برای رسم نمودار live
        total_timesteps: تعداد گام‌ها
        report_every: هر چند گام یک داده برای نمودار تولید شود
        """
        state, _ = self.env.reset()
        action = self._select_action(state)
        cumulative_reward = 0

        for t in range(1, total_timesteps + 1):
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            next_action = self._select_action(next_state)

            # SARSA update (on-policy)
            td_target = reward + self.gamma * self.q_table[next_state][next_action] * (1 - done)
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.lr * td_error

            # آماده‌سازی state و action بعدی
            if done:
                state, action = self.env.reset()[0], self._select_action(self.env.reset()[0])
            else:
                state, action = next_state, next_action

            cumulative_reward += reward

            # decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # yield برای نمودار هر report_every گام
            if t % report_every == 0:
                yield cumulative_reward, t

        return self


    # ----------------------------------------
    # Prediction (مثل SB3)
    # ----------------------------------------
    def predict(self, obs, deterministic=True):
        if deterministic:
            action = np.argmax(self.q_table[obs])
        else:
            action = self._select_action(obs)
        return action, None

    # ----------------------------------------
    # Save / Load
    # ----------------------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    @classmethod
    def load(cls, path, env):
        model = cls("MlpPolicy", env)
        with open(path, "rb") as f:
            model.q_table = pickle.load(f)
        return model
