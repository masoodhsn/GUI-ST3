import numpy as np
import pickle
import gymnasium as gym

from learning_algorithm.base_algorithm import BaseRLAlgorithm


class QLearning(BaseRLAlgorithm):

    algorithm_name = "Q-Learning"

    @classmethod
    def is_compatible(cls, env):

        return (
            isinstance(
                env.observation_space,
                gym.spaces.Discrete
            )
            and
            isinstance(
                env.action_space,
                gym.spaces.Discrete
            )
        )
    
    @classmethod
    def get_init_params(cls):

        return {
            "learning_rate": 0.1,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.995
        }



    def __init__(self, env, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay):

        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        self.q_table = np.zeros(
            (
                self.n_states,
                self.n_actions
            )
        )

    # -------------------
    # Action selection
    # -------------------
    def _select_action(self, state):

        if np.random.rand() < self.epsilon:

            return self.env.action_space.sample()

        return np.argmax(
            self.q_table[state]
        )

    # -------------------
    # Training
    # -------------------
    def learn(
        self,
        total_timesteps,
        report_every=100
    ):

        state = self._reset_env()

        cumulative_reward = 0

        for t in range(
            1,
            total_timesteps + 1
        ):

            action = self._select_action(
                state
            )

            next_state, reward, done = self._step_env(
                action
            )

            best_next_action = np.argmax(
                self.q_table[next_state]
            )

            td_target = (
                reward
                +
                self.gamma
                *
                self.q_table[next_state][best_next_action]
                *
                (1 - done)
            )

            td_error = (
                td_target
                -
                self.q_table[state][action]
            )

            self.q_table[state][action] += (
                self.lr * td_error
            )

            cumulative_reward += reward

            if done:

                state = self._reset_env()

            else:

                state = next_state

            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

            if t % report_every == 0:

                yield (
                    cumulative_reward,
                    t
                )

        return self

    # -------------------
    # Inference
    # -------------------
    def predict(
        self,
        obs,
        deterministic=True
    ):

        # VecEnv obs
        if isinstance(
            obs,
            np.ndarray
        ) and obs.shape != ():

            obs = obs[0]

        if deterministic:

            action = np.argmax(
                self.q_table[obs]
            )

        else:

            action = self._select_action(
                obs
            )

        return action, None

    # -------------------
    # Save / Load
    # -------------------
    def save(
        self,
        path
    ):

        with open(
            path,
            "wb"
        ) as f:

            pickle.dump(
                {
                    "q_table": self.q_table,
                    "epsilon": self.epsilon
                },
                f
            )

    @classmethod
    def load(
        cls,
        path,
        env
    ):

        model = cls(
            "MlpPolicy",
            env
        )

        with open(
            path,
            "rb"
        ) as f:

            data = pickle.load(f)

        model.q_table = data[
            "q_table"
        ]

        model.epsilon = data[
            "epsilon"
        ]

        return model