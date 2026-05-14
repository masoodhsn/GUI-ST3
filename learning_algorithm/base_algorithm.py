from abc import ABC, abstractmethod


class BaseRLAlgorithm(ABC):

    algorithm_name = "BaseAlgorithm"

    @classmethod
    @abstractmethod
    def get_hyperparameters(cls):
        """
        Returns default hyperparameters.

        Example:
        {
            "learning_rate": {
                "default": 0.0003,
                "type": float
            },
            "gamma": {
                "default": 0.99,
                "type": float
            }
        }


        value
        max
        min
        type
        sidebar type
        step
        """
        pass



    @classmethod
    @abstractmethod
    def is_compatible(cls, env):
        """
        Checks if this algorithm is suitable
        for the given environment.
        """
        pass


    @abstractmethod
    def export_training_state(self):
        """
        Return training state needed for continuing training.
        """
        pass


    @abstractmethod
    def import_training_state(self, state):
        """
        Restore training state.
        """
        pass


    @abstractmethod
    def predict(
        self,
        observation,
        deterministic=True
    ):
        """
        Predict next action.
        Should match SB3 API.
        """
        pass


    @abstractmethod
    def learn(
        self,
        total_timesteps,
        callback=None,
        **kwargs
    ):
        """
        Main training function.
        """
        pass


    @abstractmethod
    def save(
        self,
        path
    ):
        """
        Save complete model state.
        """
        pass


    @classmethod
    @abstractmethod
    def load(
        cls,
        path,
        env=None,
        **kwargs
    ):
        """
        Load saved model.
        """
        pass


    @staticmethod
    def get_supported_callbacks():
        """
        Optional callback config.
        Can be overridden.
        """
        return []