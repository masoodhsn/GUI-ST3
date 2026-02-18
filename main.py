import gymnasium as gym
import time
from QLearning import QLearning
import streamlit as st
from PIL import Image

st.title("CliffWalking - Q-Learning Live Chart")

# -------------------------------
# 1. محیط و مدل
# -------------------------------
if "env" not in st.session_state:
    st.session_state.env = gym.make("CliffWalking-v1", render_mode="rgb_array")

if "model" not in st.session_state:
    st.session_state.model = QLearning(
        "MlpPolicy",
        st.session_state.env,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
    )

env = st.session_state.env
model = st.session_state.model

# placeholder ها
status_text = st.empty()
display = st.empty()
reward_chart = st.line_chart([], width="stretch")  # جایگزین use_container_width

# -------------------------------
# 1. Slider برای تعداد گام‌ها
# -------------------------------
total_timesteps = st.slider(
    "Select total timesteps for training:",
    min_value=1000,
    max_value=50000,
    value=30000,
    step=1000
)

# -------------------------------
# 2. دکمه Train
# -------------------------------
if st.button("Train Model"):
    status_text.text("Training model...")
    rewards = []
    for cum_reward, step in model.learn(total_timesteps=total_timesteps, report_every=100):
        rewards.append(cum_reward)
        reward_chart.line_chart(rewards)  # نمودار live
    status_text.text("Training finished!")

# -------------------------------
# 3. دکمه Run Episode
# -------------------------------
if st.button("Run Episode"):
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

    st.success(f"Episode finished in {step} steps. Total reward: {total_reward}")
