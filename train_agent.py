import torch as th
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PlatooningEnv import PlatooningEnv  # Import the environment we created earlier

# Create a function to define and train the agent
def train_agent():

    env = PlatooningEnv()

    env = DummyVecEnv([lambda: env])

    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=20000)

    single_env = env.envs[0]

    headway_df = pd.DataFrame(single_env.headway_details1, columns=["Headway"])
    print(f"""
    Mean : {headway_df["Headway"].mean()}
    Median : {headway_df["Headway"].median()}
    Std : {headway_df["Headway"].std()}
    """)
    # Plot the line plot
    plt.figure(figsize=(15, 5))
    plt.plot(headway_df.index, headway_df["Headway"])
    plt.xlabel("Time Step")
    plt.ylabel("Headway")
    plt.title("Headway over Time")
    plt.show()

    # Save the plot
    plt.savefig("headway_plot1.png") 

    reward_df = pd.DataFrame(single_env.rewards, columns=["Reward"])

    # Plot the line plot
    plt.figure(figsize=(15, 5))
    plt.plot(reward_df.index, reward_df["Reward"])
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title("Reward over Time")
    plt.show()

    # Save the plot
    plt.savefig("reward_plot.png") 
    model.save("ppo_platooning_model")
    env.close()

# Train the agent
train_agent()
