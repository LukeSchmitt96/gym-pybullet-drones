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
from gym_pybullet_drones.utils.Logger import Logger

from gym_pybullet_drones.envs.RLTetherAviary import RLTetherAviary

############# This script will play and record a video and the performance of a trained model


if __name__ == "__main__":

    #### Initialize the logger #########################################################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS), num_drones=ARGS.num_drones)

    env = RLTetherAviary(gui=False, record=True)
    
    model_name = "./models/l_ipynb/ddpg800000"
    model = DDPG.load(model_name)

    obs = env.reset()

    done = False

    for i in range(10*env.SIM_FREQ):
        action, _     = model.predict(obs, deterministic=True)
        _, _, done, _ = env.step(action)
        if done: break

    env.close()