import streamlit as st

from backend import Backend
from frontend import Frontend


# init
frontend = Frontend()

frontend.setup_page()

# layout
display, chart = (
    frontend.create_layout()
)


# sidebar
(
    train_button,
    run_button,
    reset_button,
    train_steps,
    learning_rate,
    gamma,
    uploaded_model,
) = frontend.create_sidebar()



if "backend" not in st.session_state:

    st.session_state.backend = Backend(
        learning_rate,
        gamma
    )


backend = st.session_state.backend



# events
if train_button:

    backend.train(
        chart,
        train_steps,
        learning_rate,
        gamma
    )


if run_button:

    backend.run(
        display
    )


if reset_button:
    backend.reset_model()


if uploaded_model is not None:

    current_file = uploaded_model.name

    if (
        "loaded_model_name" not in st.session_state
        or
        st.session_state.loaded_model_name != current_file
    ):

        backend.load(uploaded_model)

        st.session_state.loaded_model_name = (
            current_file
        )