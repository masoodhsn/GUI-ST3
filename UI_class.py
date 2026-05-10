from custom_games.CartPole.env import CustomEnv
from learning_algorithm.dqn import CustomDQN


class UI():

    def __init__(self):
        self.env = CustomEnv()
        self.model = CustomDQN(
            "MlpPolicy",
            self.env,
            **CustomDQN.get_init_params(),
            verbose=1
        )
        self.last_frame = None
    
    def is_compatible(self):
        print("Algorithm compatible:",
      self.model.is_compatible(self.env))
        
    def run(self , display):
        self.model.set_env(self.env)

        obs, info = self.env.reset()
        for _ in range(1000): 
            action, _states = self.model.predict(obs, deterministic=True) 
            obs, reward, terminated, truncated, info = self.env.step(action) 
            frame = self.env.render().copy()   

            display.image(frame) 
            if terminated or truncated: 
                obs, info = self.env.reset() 
                #break 

    def train(self):
        self.model.learn(total_timesteps=10000)

    def save(self):
        self.model.save("test")

    def load(self):
        self.model = CustomDQN.load("test" , env = self.env)
