import streamlit as st

from UI_class import UI


st.set_page_config(
    page_title="RL Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(
    "Reinforcement Learning Platform"
)


# session state
if "ui" not in st.session_state:

    st.session_state.ui = UI()


ui = st.session_state.ui


# layout
col1, col2 = st.columns(2)

with col1:

    display = st.empty()

with col2:

    chart = st.empty()


# sidebar
with st.sidebar:

    train_col, run_col = st.columns(2)

    with train_col:

        train_button = st.button(
            "Train"
        )

    with run_col:

        run_button = st.button(
            "Run"
        )

    save_col, load_col = st.columns(2)

    with save_col:

        save_button = st.button(
            "Save"
        )

    with load_col:

        load_button = st.button(
            "Load"
        )


# buttons
if train_button:

    ui.train(chart)


if run_button:

    ui.run(display)


if save_button:

    ui.save()


if load_button:

    ui.load()