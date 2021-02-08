import os
import time
import pdb
import math
import numpy as np
import pybullet as p
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from gym import error, spaces, utils
from gym.utils import seeding
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.utils.Logger import Logger

from gym_pybullet_drones.envs.RLTetherAviary import RLTetherAviary

############# This script will play and record a video and the performance of a trained model


if __name__ == "__main__":

    np.random.seed(2020)

    #### Initialize the environment #########################################################################
    env = RLTetherAviary(gui=False, record=False)

    #### Initialize the logger #########################################################################
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ), num_drones=1)

    model_name = "<path-to-model>"
    # model = DDPG.load(model_name)
    # model = PPO.load(model_name)
    loaded = keras.models.model_from_json(model_name)

    obs = env.reset()

    done = False
    
    for i in range(10*env.SIM_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        logger.log(drone=0, 
                   timestamp=1/env.SIM_FREQ,
                   state=obs,
                   control=action)

        # env.render()
    
    env.close()

    # logger.save()
    # logger.plot()
    logger.plot3D("DDPG")