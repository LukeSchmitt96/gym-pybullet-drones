import os
import time
import pdb
import math
import numpy as np
import pybullet as p
import gym
import time
import datetime
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from gym_pybullet_drones.envs.RLTetherAviary import RLTetherAviary

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

if __name__ == "__main__":

    print(os.getcwd())

    #### Set up learning env and training parameters ###################################################
    log_dir = os.path.join("logs", "learn_DDPG-" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    print(log_dir)
    print(os.path.join(os.getcwd(), log_dir))
    os.makedirs(log_dir)
    step_iters = 10
    training_timesteps = 500000

    #### Create custom policy ##########################################################################
    CustomPolicy = MlpPolicy
    CustomPolicy.layers = [64,64,32]    # actor network has layers [64, 64, 32]

    #### Check the environment's spaces ################################################################
    env = RLTetherAviary(gui=False, record=False)
    env = Monitor(env, log_dir)
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)
    print("[INFO] Checking Environment...")
    check_env(env, warn=True, skip_render_check=True) 

    ####
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.N_ACTIONS), sigma=0.1 * np.ones(env.N_ACTIONS), dt = 0.005)
    
    
    #### Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    #### Train the model ###############################################################################
    model = DDPG(CustomPolicy, env, verbose=1, batch_size=64, action_noise=action_noise)
    
    for i in range(step_iters):     # run for step_iters * training_timesteps

        model.learn(total_timesteps=training_timesteps)

        model.save("./models/ddpg"+str((i+1)*training_timesteps))
        model.save_replay_buffer("./experiences/ddpg_experience"+str((i+1)*training_timesteps))

        #### Show (and record a video of) the model's performance ##########################################
        env_test = RLTetherAviary(gui=False, record=True)
        obs = env_test.reset()
        start = time.time()
        for i in range(10*env_test.SIM_FREQ):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_test.step(action)
            if done: break
        env_test.close()

    env.close()
    
    results_plotter.plot_results([os.path.join(os.getcwd(), log_dir)], step_iters * training_timesteps, results_plotter.X_TIMESTEPS, "DDPG")
    
    plot_results(os.path.join(os.getcwd(), log_dir), "DDPG")