import numpy as np
import pickle
import gymnasium as gym


class QLearning:
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
        """
        Q-Learning implementation compatible with:
        - Gymnasium environments
        - Stable-Baselines3 DummyVecEnv (n_envs=1)
        """

        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.verbose = verbose

        # Support both Gym Env and VecEnv
        obs_space = env.observation_space
        act_space = env.action_space

        assert isinstance(obs_space, gym.spaces.Discrete)
        assert isinstance(act_space, gym.spaces.Discrete)

        self.n_states = obs_space.n
        self.n_actions = act_space.n

        # Initialize Q-table
        self.q_table = np.zeros((self.n_states, self.n_actions))

    # ----------------------------------------
    # Handle reset for both Gym and VecEnv
    # ----------------------------------------
    def _reset_env(self):
        obs = self.env.reset()

        # VecEnv returns only obs (as array)
        if isinstance(obs, np.ndarray):
            return obs[0]

        # Gym returns (obs, info)
        return obs[0]

    # ----------------------------------------
    # Handle step for both Gym and VecEnv
    # ----------------------------------------
    def _step_env(self, action):
        # If VecEnv, action must be a list
        if hasattr(self.env, "num_envs"):
            obs, reward, done, info = self.env.step([action])
            return obs[0], reward[0], done[0]
        else:
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            return obs, reward, done

    # ----------------------------------------
    # ε-greedy action selection
    # ----------------------------------------
    def _select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    # ----------------------------------------
    # Training loop
    # ----------------------------------------
    def learn(self, total_timesteps, report_every=100):
        """
        Train Q-Learning and yield cumulative reward periodically.
        """

        state = self._reset_env()
        cumulative_reward = 0

        for t in range(1, total_timesteps + 1):

            action = self._select_action(state)
            next_state, reward, done = self._step_env(action)

            # Q-Learning update (off-policy)
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (1 - done)
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.lr * td_error

            state = next_state if not done else self._reset_env()
            cumulative_reward += reward

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if t % report_every == 0:
                yield cumulative_reward, t

        return self

    # ----------------------------------------
    # Prediction (SB3-like API)
    # ----------------------------------------
    def predict(self, obs, deterministic=True):

        # If observation is batched (VecEnv), extract first element
        if isinstance(obs, np.ndarray) and obs.shape != ():
            obs = obs[0]

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