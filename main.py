#model.save("dqn_model", include=["replay_buffer"])
#model.save("dqn_model", include_replay_buffer=True)
#model = DQN.load("dqn_model", env=env)

# save models
# load models

# multi-threading
# seperate ui and sb3(back- end)
# create a madole to impliment learning algorithm

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

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from learning_algorithm.base_algorithm import BaseRLAlgorithm
from callbacks.training_callback import StreamlitTrainingCallback


# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="RL Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Reinforcement Learning Platform")


# -------------------------------
# Load environment dynamically
# -------------------------------
def load_custom_env(game_folder):

    env_path = os.path.join(
        "custom_games",
        game_folder,
        "env.py"
    )

    spec = importlib.util.spec_from_file_location(
        "custom_env",
        env_path
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    env_class = getattr(
        module,
        "CustomEnv"
    )

    env = env_class()

    if hasattr(env, "render_mode"):
        env.render_mode = "rgb_array"

    return env


def make_env(game):

    env = load_custom_env(game)
    env = Monitor(env)
    return env


# -------------------------------
# Load algorithms dynamically
# -------------------------------
def load_algorithms():

    algorithms = {}

    folder = "learning_algorithm"

    for file in os.listdir(folder):

        if (
            file.endswith(".py")
            and file not in ["__init__.py", "base_algorithm.py"]
        ):

            path = os.path.join(folder, file)

            spec = importlib.util.spec_from_file_location(
                file[:-3],
                path
            )

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name in dir(module):

                obj = getattr(module, name)

                if (
                    isinstance(obj, type)
                    and hasattr(obj, "is_compatible")
                    and issubclass(obj, BaseRLAlgorithm)
                    and obj != BaseRLAlgorithm
                ):
                    algorithms[obj.algorithm_name] = obj

    return algorithms


# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:

    # games
    available_games = []

    if os.path.exists("custom_games"):
        available_games = [
            f for f in os.listdir("custom_games")
            if os.path.isdir(os.path.join("custom_games", f))
        ]

    selected_game = st.selectbox(
        "Select Game",
        available_games
    )

    # algorithms
    algorithm_classes = load_algorithms()

    selected_algo = None

    if selected_game:

        temp_env = load_custom_env(selected_game)

        algo_labels = []

        for name, cls in algorithm_classes.items():

            ok = cls.is_compatible(temp_env)

            label = name + (" 🟢 reco" if ok else " 🔴")

            algo_labels.append((label, name))

        selected_label = st.selectbox(
            "Select Algorithm",
            [x[0] for x in algo_labels]
        )

        for label, name in algo_labels:
            if label == selected_label:
                selected_algo = name
                break

    # buttons
    col1, col2 = st.columns(2)

    with col1:
        train_button = st.button("Train")

    with col2:
        run_button = st.button("Run")

    total_timesteps = st.slider(
        "Timesteps",
        1000,
        50000,
        20000,
        1000
    )

    with st.expander("Hyperparameters"):
        AlgoClass = algorithm_classes[selected_algo]
        params = AlgoClass.get_init_params()

        for key, value in params.items():

            params[key] = st.slider(
                key,
                0.0,
                1.0,
                float(value)
            )


# -------------------------------
# Session state
# -------------------------------
if selected_game:

    if (
        "game" not in st.session_state
        or st.session_state.game != selected_game
    ):

        st.session_state.game = selected_game

        st.session_state.env = make_env(selected_game)

        st.session_state.vec_env = DummyVecEnv([
            lambda: make_env(selected_game)
        ])

        st.session_state.model = None


env = st.session_state.get("env", None)
model = st.session_state.get("model", None)


# -------------------------------
# UI
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    display = st.empty()

with col2:
    chart = st.line_chart([])

status = st.empty()


# -------------------------------
# Train
# -------------------------------
if train_button and env and selected_algo:

    AlgoClass = algorithm_classes[selected_algo]

    params = AlgoClass.get_init_params()

    st.session_state.model = AlgoClass.from_params(
        env=env,
        params=params
    )

    model = st.session_state.model

    # environment for live rendering
    render_env = make_env(selected_game)

    # callback
    callback = StreamlitTrainingCallback(
        render_env=render_env,
        display_placeholder=display,
        chart_placeholder=chart,
        render_every=500,
        reward_window=100
    )

    status.text("Training...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    status.success("Training finished")


# -------------------------------
# Run
# -------------------------------
if run_button and env and model:

    obs = env.reset()[0]

    done = False

    total_reward = 0

    steps = 0

    while not done:

        action, _ = model.predict(obs)

        action = int(action)

        obs, r, term, trunc, _ = env.step(action)

        done = term or trunc

        total_reward += r

        steps += 1

        display.image(Image.fromarray(env.render()))

        time.sleep(0.1)

    status.success(
        f"Done | steps={steps} | reward={total_reward}"
    )