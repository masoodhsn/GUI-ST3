import streamlit as st


################### show logs   


class Frontend:

    def setup_page(self):

        st.set_page_config(
            page_title="RL Dashboard",
            page_icon="🎮",
            layout="wide"
        )

        st.title(
            "Reinforcement Learning Platform"
        )


    def create_layout(self):

        col1, col2 = st.columns(2)

        with col1:
            display = st.empty()

        with col2:
            chart = st.empty()

        return display, chart


    def create_sidebar(self):

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


            reset_button = st.button(
                "Reset Model"
            )

            uploaded_model = st.file_uploader(
                "Load Model",
                type=["zip"]
            )

            train_steps = st.slider(
                "Train Steps",
                min_value=1000,
                max_value=1_000_000,
                value=70_000,
                step=1000
                )

            with st.expander("⚙️ Hyperparameters"):

                learning_rate = st.number_input(
                    "Learning Rate",
                    value=0.0003,
                    format="%.5f"
                )

                gamma = st.slider(
                    "Gamma",
                    value=0.99,
                    format="%.2f"
                )

        return (
            train_button,
            run_button,
            reset_button,
            train_steps,
            learning_rate,
            gamma,
            uploaded_model      
        )