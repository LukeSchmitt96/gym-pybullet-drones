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
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.RLTetherAviary import RLTetherAviary

if __name__ == "__main__":

    env = RLTetherAviary(gui=1, record=0)
    
    model_name = "./models/v/ddpg1600000"
    model = DDPG.load(model_name)


    obs = env.reset()

    for i in range(100000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)