#model.save("dqn_model", include=["replay_buffer"])
#model.save("dqn_model", include_replay_buffer=True)
#model = DQN.load("dqn_model", env=env)

# save models
# load models

# Sandbox execution( Docker container,Python sandbox libraries, Process-level sandbox)
# multi-threading
# seperate ui and sb3(back- end)

# reset()
# step(action)
# render()
# close()
# observation_space
# action_space

# Custom environments must support rgb_array rendering
import os
import importlib.util
import gymnasium as gym
import time
import streamlit as st
from PIL import Image

from QLearning import QLearning
from sarsa import SARSA

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="RL Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Reinforcement Learning Platform")


# -------------------------------
# Dynamic environment loader
# -------------------------------
def load_custom_env(game_folder):
    """
    Dynamically load CustomEnv class from:
    custom_games/<game_folder>/env.py
    """

    env_path = os.path.join("custom_games", game_folder, "env.py")

    spec = importlib.util.spec_from_file_location("custom_env", env_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    env_class = getattr(module, "CustomEnv")
    env = env_class()

    # Ensure rgb_array rendering is supported
    if hasattr(env, "render_mode"):
        env.render_mode = "rgb_array"

    return env


# -------------------------------
# Environment factory
# -------------------------------
def make_env(selected_game):
    """
    Create monitored environment from selected custom game.
    """
    env = load_custom_env(selected_game)
    env = Monitor(env)
    return env


# -------------------------------
# Sidebar controls
# -------------------------------
with st.sidebar:

    # List available games inside custom_games folder
    if os.path.exists("custom_games"):
        available_games = [
            folder for folder in os.listdir("custom_games")
            if os.path.isdir(os.path.join("custom_games", folder))
        ]
    else:
        available_games = []

    selected_game = st.selectbox("Select Game", available_games)

    col1, col2 = st.columns(2)

    with col1:
        train_button = st.button("Train Model")
    with col2:
        run_button = st.button("Run Episode")

    total_timesteps = st.slider(
        "Total Timesteps",
        min_value=1000,
        max_value=50000,
        value=30000,
        step=1000
    )

    with st.expander("Advanced Settings"):
        lr = st.slider("Learning Rate", 0.01, 1.0, 0.5, 0.01)
        gamma = st.slider("Gamma", 0.1, 0.99, 0.99, 0.01)
        epsilon = st.slider("Epsilon", 0.1, 0.99, 1.0, 0.01)
        epsilon_decay = st.slider("Epsilon Decay", 0.1, 0.995, 0.995, 0.005)
        epsilon_min = st.slider("Minimum Epsilon", 0.01, 0.99, 0.05, 0.01)


# -------------------------------
# Session state initialization
# -------------------------------
if selected_game:

    if "env" not in st.session_state or st.session_state.get("current_game") != selected_game:
        st.session_state.current_game = selected_game
        st.session_state.env = make_env(selected_game)
        st.session_state.vec_env = DummyVecEnv([lambda: make_env(selected_game)])

        st.session_state.model = QLearning(
            "MlpPolicy",
            st.session_state.env,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay
        )


env = st.session_state.get("env", None)
vec_env = st.session_state.get("vec_env", None)
model = st.session_state.get("model", None)


# -------------------------------
# UI Layout
# -------------------------------
col_left, col_right = st.columns(2)

with col_left:
    display = st.empty()

with col_right:
    reward_chart = st.line_chart([], width='stretch')

status_text = st.empty()


# -------------------------------
# Training Section
# -------------------------------
if train_button and env is not None:

    status_text.text("Training model...")

    st.session_state.model = QLearning(
        "MlpPolicy",
        env,
        learning_rate=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )

    model = st.session_state.model

    render_env = make_env(selected_game)
    obs = render_env.reset()[0]

    cumulative_rewards = []

    for cum_reward, step in model.learn(
        total_timesteps=total_timesteps,
        report_every=100
    ):

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = render_env.step(action)

        done = terminated or truncated
        if done:
            obs = render_env.reset()[0]

        frame = render_env.render()
        img = Image.fromarray(frame)
        display.image(img)

        if step % 1000 == 0:
            cumulative_rewards.append(cum_reward)
            reward_chart.line_chart(cumulative_rewards)

        time.sleep(0.2)

    status_text.text("Training finished!")


# -------------------------------
# Run Episode Section
# -------------------------------
if run_button and env is not None:

    obs = env.reset()[0]
    done = False
    total_reward = 0
    steps = 0

    while not done:

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        total_reward += reward
        steps += 1

        frame = env.render()
        img = Image.fromarray(frame)
        display.image(img)

        time.sleep(0.2)

    status_text.success(
        f"Episode finished in {steps} steps | Total Reward: {total_reward}"
    )