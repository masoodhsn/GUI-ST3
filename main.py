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
    uploaded_model,
) = frontend.create_sidebar()


algorithm = frontend.RL_setup(Backend.get_available_algorithms())

if "backend" not in st.session_state:

    st.session_state.backend = Backend(
        algorithm
    )
backend = st.session_state.backend




if algorithm is not None:

    current_algorithm = algorithm

    if (
        "algorithm" not in st.session_state
        or
        st.session_state.algorithm != current_algorithm
    ):

        backend.set_algorithm(algorithm)

        st.session_state.algorithm = (
            current_algorithm
        )



hyperparams = frontend.hyperparametrs(backend.hyperparams)



if hyperparams is not None:

    current_hyperparams = hyperparams

    if (
        "hyperparams" not in st.session_state
        or
        st.session_state.hyperparams != current_hyperparams
    ):

        backend.update_hyperparams(hyperparams)

        st.session_state.hyperparams = (
            current_hyperparams
        )


# events
if train_button:

    backend.train(
        chart,
        train_steps,
    )


if run_button:

    backend.run(
        display
    )


if reset_button:
    backend.build_model()


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
