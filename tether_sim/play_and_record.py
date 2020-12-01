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

############# This script will play and record a model

if __name__ == "__main__":

    env = RLTetherAviary(gui=True, record=True)
    
    model_name = "./models/"
    model = DDPG.load(model_name)

    obs = env.reset()

    done = False

    while ~done:
        action, _     = model.predict(obs, deterministic=True)
        _, _, done, _ = env.step(action)