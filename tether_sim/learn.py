import os
import time
import pdb
import math
import numpy as np
import pybullet as p
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env

from gym_pybullet_drones.envs.RLTetherAviary import RLTetherAviary

if __name__ == "__main__":

    #### Check the environment's spaces ################################################################
    #env = gym.make("rl-CrazyFlie-aviary-v0")
    env = RLTetherAviary(gui=True, record=False)
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)
    print("[INFO] Checking Environment...")
    check_env(env, warn=True, skip_render_check=True) 

    #### Train the model ###############################################################################
    model = DDPG(MlpPolicy, env, verbose=1, batch_size=64)
    
    training_timesteps = 500000
    
    for i in range(10):

        model.learn(total_timesteps=training_timesteps)
        model.save("ddpg"+str((i+1)*training_timesteps))
        model.save_replay_buffer("ddpg_experience"+str((i+1)*training_timesteps))

        #### Show (and record a video of) the model's performance ##########################################
        env_test = RLTetherAviary(gui=1, record=0)
        obs = env_test.reset()
        start = time.time()
        for i in range(10*env_test.SIM_FREQ):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_test.step(action)
            env_test.render()
            if done: break
        env_test.close()

    env.close()