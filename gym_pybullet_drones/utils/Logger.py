import os
import time
from datetime import datetime
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


######################################################################################################################################################
#### Logger class ####################################################################################################################################
######################################################################################################################################################
class Logger(object):

    ####################################################################################################
    #### Initialize the logger #########################################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - logging_freq_hz (int)            logging frequency in Hz ####################################
    #### - num_drones (int)                 number of drones ###########################################
    #### - duration_sec (int)               (opt) to preallocate the log arrays, improves performance ##
    ####################################################################################################
    def __init__(self, logging_freq_hz: int, num_drones: int=1, duration_sec: int=0):
        self.LOGGING_FREQ_HZ = logging_freq_hz; self.NUM_DRONES = num_drones
        self.PREALLOCATED_ARRAYS = False if duration_sec==0 else True
        self.counters   = np.zeros(num_drones)
        self.timestamps = np.zeros((num_drones,     duration_sec*self.LOGGING_FREQ_HZ))
        self.states     = np.zeros((num_drones, 16, duration_sec*self.LOGGING_FREQ_HZ))
        self.controls   = np.zeros((num_drones, 5,  duration_sec*self.LOGGING_FREQ_HZ))

    ####################################################################################################
    #### Log entries for a single simulation step, of a single drone ###################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - drone (int)                      drone associated to the log entry ##########################
    #### - timestamp (float)                timestamp of the log in simulation clock ###################
    #### - state ((20,1) array)             drone's state ##############################################
    #### - control ((12,1) array)           drone's control target #####################################
    ####################################################################################################
    def log(self, drone: int, timestamp, state, control):
        current_counter = int(self.counters[drone])
        #### Add rows to the logging matrices every time a counter exceeds their shape #####################
        if current_counter>=self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps,  np.zeros((self.NUM_DRONES, 1))),     axis=1)
            self.states     = np.concatenate((self.states,      np.zeros((self.NUM_DRONES, 16, 1))), axis=2)
            self.controls   = np.concatenate((self.controls,    np.zeros((self.NUM_DRONES, 5, 1))),  axis=2)
        #### Advance a counter is the logging matrices have grown faster than it has #######################
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1]>current_counter: current_counter = self.timestamps.shape[1]-1
        #### Log the information and increase the counter ##################################################
        self.timestamps[drone,current_counter] = timestamp
        self.states[drone,:,current_counter]   = np.hstack([ state ])
        self.controls[drone,:,current_counter] = np.hstack([ control ])
        self.counters[drone] = current_counter + 1

    ####################################################################################################
    #### Save the logs to file #########################################################################
    ####################################################################################################
    def save(self):
        with open(os.path.dirname(os.path.abspath(__file__))+"/../../files/save-flight-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".npy", 'wb') as out_file:
            np.savetxt(out_file, (self.timestamps, self.states, self.controls), delimiter=',')

    ####################################################################################################
    #### Plot the logged (state) data ##################################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - pwm (bool)                       if True, convert logged RPM into PWM values ################
    ####################################################################################################
    def plot(self, pwm=False):
        #### Loop over colors and line styles ##############################################################
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))  
        fig, axs = plt.subplots(6,2)
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)
        labels = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'R', 'P', 'Yaw', 'WR', 'WP', 'WY'] if pwm else ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'R', 'P', 'Yaw', 'WR', 'WP', 'WY']
        for i in range(len(labels)):
            for j in range(self.NUM_DRONES):
                #### This line converts RPM into PWM for all drones except drone_0 (used in compare.py) ############
                if pwm and i>11 and j>0: self.states[j,i,:] = (self.states[j,i,:] - 4070.3) / 0.2685
                axs[i%6,i//6].plot(t, self.states[j,i,:], label="drone_"+str(j))
            axs[i%6,i//6].set_xlabel('time')
            axs[i%6,i//6].set_ylabel(labels[i])
            axs[i%6,i//6].grid(True)
        fig.subplots_adjust(left=0.06, bottom=0.05, right=0.99, top=0.98, wspace=0.15, hspace=0.0)
        plt.show()

    def plot3D(self, method):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.states[0,0,:], self.states[0,1,:], self.states[0,2,:])

        plt.xlim([-0.25, 0.25])
        plt.ylim([-0.25, 0.25])
        ax.set_zlim(0, 1.25)

        plt.title(f"Flight Using {method} Method")

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')

        plt.savefig(f"/home/luke/Desktop/MLAI/plots/{method}.png")

        plt.show()

