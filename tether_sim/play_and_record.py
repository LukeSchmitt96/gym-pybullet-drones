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

    np.random.seed(24787)

    #### Initialize the environment #########################################################################
    env = RLTetherAviary(gui=True, record=False)

    #### Initialize the logger #########################################################################
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ), num_drones=1)

    # models/ddpg3000000.zip
    model_name = "/home/wrobotics11/gym-pybullet-drones/models/ddpg3000000"
    model = DDPG.load(model_name)

    obs = env.reset()

    done = False

    # for i in range(10*env.SIM_FREQ):
    #     action, _     = model.predict(obs, deterministic=True)
    #     _, _, done, _ = env.step(action)
    #     if done: break
    
    for i in range(10*env.SIM_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        logger.log(drone=0, 
                   timestamp=1/env.SIM_FREQ,
                   state=obs)

        # env.render()
    
    env.close()

    logger.save()
    logger.plot()