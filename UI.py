import streamlit as st
from UI_class import UI

st.set_page_config(
    page_title="RL Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Reinforcement Learning Platform")

if "ui" not in st.session_state:
    st.session_state.ui = UI()

ui = st.session_state.ui

with st.sidebar:
    train, run = st.columns(2)

    with train:
        train_button = st.button("Train")

    with run:
        run_button = st.button("Run")

    save, load = st.columns(2)

    with save:
        save_button = st.button("Save")

    with load:
        load_button = st.button("Load")


if train_button:
    ui.train()

if run_button:
    display = st.empty()
    ui.run(display)

if save_button:
    ui.save()

if load_button:
    ui.load()

