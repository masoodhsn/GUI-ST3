import gymnasium as gym
import time
from QLearning import QLearning
from sarsa import SARSA
import streamlit as st
from PIL import Image


#model.save("dqn_model", include=["replay_buffer"])
#model.save("dqn_model", include_replay_buffer=True)
#model = DQN.load("dqn_model", env=env)

# save models
# load models



st.set_page_config(
    page_title="RL Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Reinforcment Learning')

with st.sidebar:
    s_col1, s_col2 = st.columns(2)

    with s_col1:
        train = st.button("Train Model")
    with s_col2:
        run = st.button("Run Episode")

    total_timesteps = st.slider(
            "Select total timesteps for training:",
            min_value=1000,
            max_value=50000,
            value=30000,
            step=1000
            )
    with st.expander("advance setting"):
        lr = st.slider("Learning Rate", 0.01, 1.0, 1.0, 0.01)
        gamma = st.slider("Gamma", 0.1, 0.99, 0.99, 0.01)
        epsilon= st.slider("Epsilon", 0.1, 0.99, 1.0, 0.01)
        epsilon_decay= st.slider("Epsilon decay", 0.1, 0.995, 0.995, 0.005)
        epsilon_min= st.slider("Minimum epsilon", 0.01, 0.99, 0.05, 0.01)
        
        
# -------------------------------
# 1. محیط و مدل
# -------------------------------
if "env" not in st.session_state:
    st.session_state.env = gym.make("CliffWalking-v1", render_mode="rgb_array")

if "model" not in st.session_state:
    st.session_state.model = QLearning(
        "MlpPolicy",
        st.session_state.env,
        learning_rate=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )


col1, col2 = st.columns(2)
with col1:
    display = st.empty()
with col2:    
    reward_chart = st.line_chart([], width="stretch")  

env = st.session_state.env
model = st.session_state.model
status_text = st.empty()


# -------------------------------
# 2. دکمه Train
# -------------------------------
if train:
    status_text.text("Training model...")

    st.session_state.model = QLearning(
        "MlpPolicy",
        st.session_state.env,
        learning_rate=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )

    model = st.session_state.model 

    rewards = [0]
    env_t = gym.make("CliffWalking-v1", render_mode="rgb_array")
    obs, _ = env_t.reset()

    for cum_reward, step in model.learn(
        total_timesteps=total_timesteps,
        report_every=100
    ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_t.step(action)
        done = terminated or truncated
        if done:
            obs, _ = env_t.reset()

        frame = env_t.render()
        img = Image.fromarray(frame)
        display.image(img)
        time.sleep(0.1)

        if step % 1000 == 0:
            rewards.append(cum_reward)
            reward_chart.line_chart(rewards)

    status_text.text("Training finished!")

# -------------------------------
# 3. دکمه Run Episode
# -------------------------------

if run:
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        # نمایش محیط گرافیکی
        frame = env.render()
        img = Image.fromarray(frame)
        display.image(img)

        time.sleep(0.1)

    #st.success(f"Episode finished in {step} steps. Total reward: {total_reward}")
