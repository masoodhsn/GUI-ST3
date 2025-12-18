# To Run (in cmd) : streamlit run gui_app.py
# gui_app.py

import streamlit as st
import time
import random
import plotly.graph_objects as go

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="RL Training Monitor",
    layout="wide"
)

# ----------------------------
# Sidebar (Controls)
# ----------------------------
st.sidebar.title("Controls")

env_name = st.sidebar.selectbox(
    "Environment",
    ["CartPole-v1", "LunarLander-v2"]
)

algo_name = st.sidebar.selectbox(
    "Algorithm",
    ["PPO", "DQN", "A2C"]
)

start_btn = st.sidebar.button("Start Training")
stop_btn = st.sidebar.button("Stop")

st.sidebar.markdown("---")
st.sidebar.text("Status: Idle")

# ----------------------------
# Main Layout
# ----------------------------
st.title("Reinforcement Learning Training Dashboard")

col1, col2 = st.columns(2)

# ----------------------------
# Live Chart Placeholder
# ----------------------------
with col1:
    st.subheader("Reward Curve")

    chart_placeholder = st.empty()

# ----------------------------
# Environment View Placeholder
# ----------------------------
with col2:
    st.subheader("Environment Execution")

    env_placeholder = st.empty()
    env_placeholder.text("Environment rendering will appear here")

# ----------------------------
# Fake Live Data (for now)
# ----------------------------
if start_btn:
    rewards = []
    steps = []

    for step in range(100):
        rewards.append(random.uniform(-10, 10))
        steps.append(step)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=steps, y=rewards, mode="lines", name="Reward")
        )
        fig.update_layout(
            xaxis_title="Step",
            yaxis_title="Reward",
            height=400
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)
        env_placeholder.text(f"Training step: {step}")

        time.sleep(0.1)
