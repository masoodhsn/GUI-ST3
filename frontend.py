import streamlit as st


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

        return (
            train_button,
            run_button,
            reset_button,
            train_steps,
            uploaded_model    
        )
    


    def RL_setup(
              self,
              algorithms
            ):
        with st.sidebar:
            with st.expander(
                    "🎮 RL Setup    "
                ):

                    algorithm = st.selectbox(
                        "🧠 Algorithms",
                        algorithms
                    )

        return algorithm 
    


    def hyperparametrs(
            self,
            hyperparametrs
        ):

        new_hyperparametrs = {}

        with st.sidebar:

            with st.expander("⚙️ Hyperparameters"):

                for param_name, param_config in hyperparametrs.items():

                    default_value = param_config["default"]
                    param_type = param_config["type"]

                    # float parameters
                    if param_type == float:

                        value = st.number_input(
                            label=param_name,
                            value=float(default_value),
                            step=0.0001,
                            format="%.6f",
                            key=f"hp_{param_name}"
                        )

                    # int parameters
                    elif param_type == int:

                        value = st.number_input(
                            label=param_name,
                            value=int(default_value),
                            step=1,
                            key=f"hp_{param_name}"
                        )

                    # fallback
                    else:
                        value = default_value

                    new_hyperparametrs[param_name] = value

        return new_hyperparametrs