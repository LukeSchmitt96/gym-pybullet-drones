import numpy as np
import time
import pybullet as p
import pybullet_data
import os
import random
from gym import error, spaces, utils

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary


######################################################################################################################################################
#### Single tethered drone environment class for reinforcement learning applications (in this implementation, at some pose) ##########################
######################################################################################################################################################
class RLTetherAviary(BaseAviary):

    ####################################################################################################
    #### Initialize the environment ####################################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - drone_model (DroneModel)         desired drone type (associated to an .urdf file) ###########
    #### - num_drones (int)                 desired number of drones in the aviary #####################
    #### - visibility_radius (float)        used to compute the drones' adjacency matrix, in meters ####
    #### - initial_xyzs ((3,1) array)       initial XYZ position of the drones #########################
    #### - initial_rpys ((3,1) array)       initial orientations of the drones (radians) ###############
    #### - physics (Physics)                desired implementation of physics/dynamics #################
    #### - freq (int)                       the frequency (Hz) at which the physics engine advances ####
    #### - aggregate_phy_steps (int)        number of physics updates within one call of .step() #######
    #### - gui (bool)                       whether to use PyBullet's GUI ##############################
    #### - record (bool)                    whether to save a video of the simulation ##################
    #### - obstacles (bool)                 whether to add obstacles to the simulation #################
    ####################################################################################################
    def __init__(self, drone_model: DroneModel=DroneModel.CERLAB, num_drones: int=1,
                        neighbourhood_radius: float=np.inf, initial_xyzs=None, initial_rpys=None,
                        physics: Physics=Physics.PYB_TETHER, freq: int=200, aggregate_phy_steps: int=1,
                        gui=False, record=False, obstacles=False, user_debug_gui=True):
        if num_drones!=1: print("[ERROR] in RLTakeoffAviary.__init__(), RLTakeoffAviary only accepts num_drones=1" ); exit()
        super().__init__(drone_model=drone_model, neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs, initial_rpys=initial_rpys, physics=physics, freq=freq, aggregate_phy_steps=aggregate_phy_steps,
            gui=gui, record=record, obstacles=obstacles, user_debug_gui=user_debug_gui) 

        self.N_ACTIONS = 5
        
        self.INIT_X_BOUND = 0.2
        self.INIT_Y_BOUND = 0.2
        self.INIT_Z_BOUND = 0.2
        self.INIT_R_BOUND = 0.1
        self.INIT_P_BOUND = 0.1
        self.INIT_YAW_BOUND = 0

        self.geoFenceMax_XY = 0.99
        self.geoFenceMax_Z  = 1.5
        self.geoFenceMin_Z  = 0
        
        self.penaltyPosition = 10
        self.penaltyAngle = 10
        self.penaltyVelocity = 50
        self.penaltyAngularVelocity = 50
        self.penaltyFlag = 1000
        self.penaltyTetherUsage = 0
        self.penaltyDiffTetherUsage = 5
        
        self.positionThreshold = 0.15
        self.angleThreshold = np.pi/18
        self.angularVelocityThreshold = 0.05
        self.velocityThreshold = 0.1
        
        self.rewardGoal = 20

        self.TETHER_MIN_LENGTH = 0.20
        self.MAX_TETHER_FORCE = 0.25*(4*(self.MAX_RPM**2 * self.KF))
        print("[INFO] Max tether force set to ", round(self.MAX_TETHER_FORCE,4), "N")
        
    def _housekeeping(self):
        #### Initialize/reset counters and zero-valued variables ###########################################
        self.RESET_TIME = time.time(); self.step_counter = 0; self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES); self.Y_AX = -1*np.ones(self.NUM_DRONES); self.Z_AX = -1*np.ones(self.NUM_DRONES)
        if self.PHYSICS == Physics.PYB_TETHER:
            self.TETHER_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES); self.USE_GUI_RPM=False; self.last_input_switch = 0
        self.last_action = -1*np.ones((self.NUM_DRONES,self.N_ACTIONS))
        self.last_clipped_action = np.zeros((self.NUM_DRONES,self.N_ACTIONS)); self.gui_input = np.zeros(self.N_ACTIONS)
        self.no_pybullet_dyn_accs = np.zeros((self.NUM_DRONES,3))
        
        #### Initialize the drones kinematic information ##################################################
        unitVector = np.random.rand(3)-np.array([0.5,0.5,0])
        startingPoint = np.ones((self.NUM_DRONES,3))*np.array([0,0,0.5])
        self.pos = 0.5*np.random.rand(1)*unitVector/np.linalg.norm(unitVector)*np.ones((self.NUM_DRONES,3))+startingPoint

        self.quat = np.zeros((self.NUM_DRONES,4)); self.rpy = np.zeros((self.NUM_DRONES,3))
        self.vel = np.zeros((self.NUM_DRONES,3)); self.ang_v = np.zeros((self.NUM_DRONES,3))

        #### Set PyBullet's parameters #####################################################################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #################################################
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        self.DRONE_IDS = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.URDF, self._setInitialXYZ(self.INIT_XYZS[i,:]), p.getQuaternionFromEuler(self._setInitialRPY(self.INIT_RPYS[i,:])), physicsClientId=self.CLIENT) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            #### Show the frame of reference of the drone, note thet it can severly slow down the GUI ##########
            if self.GUI and self.USER_DEBUG: self._showDroneLocalAxes(i)
            #### Disable collisions between drones' and the ground plane, e.g., to start a drone at [0,0,0] ####
            p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES: self._addObstacles()


    ####################################################################################################
    #### Return the action space of the environment, a Box(4,) #########################################
    ####################################################################################################
    #### Additional component in action space for tether force magnitude ###############################
    ####################################################################################################
    def _actionSpace(self):
        #### Action vector ######## P0            P1            P2            P3            P4
        act_lower_bound = np.array([-1,           -1,           -1,           -1,           0])
        act_upper_bound = np.array([ 1,            1,            1,            1,           1])   
        return spaces.Box( low=act_lower_bound, high=act_upper_bound, dtype=np.float32 )
        
    ####################################################################################################
    #### Return the observation space of the environment, a Box(20,) ###################################
    ####################################################################################################
    def _observationSpace(self):
        #### Observation vector ### X        Y        Z      R       P       Y       VX       VY       VZ       WR       WP       WY       P0            P1            P2            P3
        obs_lower_bound = np.array([-1,      -1,      0,     -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
        obs_upper_bound = np.array([ 1,       1,      1,      1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])
        return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )

    ####################################################################################################
    #### Return the current observation of the environment #############################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - obs (20,) array                  for its content see _observationSpace() ####################
    ####################################################################################################
    def _computeObs(self):
        droneState = self._getDroneStateVector(0)
        droneState = np.hstack([droneState[0:3],droneState[7:20]])
        droneState.reshape(16,)        
        return self._clipAndNormalizeState(droneState)

    ####################################################################################################
    #### Preprocess the action passed to step() ########################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - action ((N_ACTIONS,1) array)     unclipped RPMs commanded to the 4 motors of the drone ######
    ####################################################################################################
    #### Returns #######################################################################################
    #### - clipped_action ((N_ACTIONS,1) array)  clipped RPMs commanded to the 4 motors of the drone ###
    ####################################################################################################
    def _preprocessAction(self, action):
        rpm, mag = self._normalizedActionToRPM(action)
        return np.hstack((np.clip(np.array(rpm), 0, self.MAX_RPM), np.clip(np.array(mag), 0, self.MAX_TETHER_FORCE)))

    ####################################################################################################
    #### Compute the current reward value(s) ###########################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - obs (..)                         the return of _computeObs() ################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - reward (..)                      the reward(s) associated to the current obs/state ##########
    ####################################################################################################
    def _computeReward(self, obs):
        # x = obs[0]
        # y = obs[1]
        # z = obs[2]

        # dx = obs[3]
        # dy = obs[4]
        # dz = obs[5]

        # roll = obs[6]
        # pitch = obs[7]
        # yaw = obs[8]

        # droll = obs[9]
        # dpitch = obs[10]
        # dyaw = obs[11]

        xy = obs[0:2]
        rp = obs[3:5]
        v = obs[6:9]
        # angle = obs[3:6]
        omega = obs[9:12]
        # actions = obs[12:16]
        # tetherUsage = self.tether_force_mag
        deltaTetherUsage = np.abs(self.tether_force_mag - self.last_tether_force_mag)


        errorPosition = np.sum(np.square(xy))                                       # error in x and y
        errorAngle = np.sum(np.square(rp))                                          # error in r and p
        errorVelocity = np.sum(np.square(v))                                        # error in dx, dy, dz
        errorAngularVelocity = np.sum(np.square(omega[0:3]))                        # error in droll, dpitch, dyaw
        
        penaltyPosition = errorPosition*self.penaltyPosition                        # get cost from position        [error * penalty]
        penaltyAngle = errorAngle*self.penaltyAngle                                 # get cost from angle           [error^2 * penalty]
        penaltyVelocity = errorVelocity*self.penaltyVelocity                        # get cost from velovity        [error * penalty]
        penaltyAngularVelocity = errorAngularVelocity*self.penaltyAngularVelocity   # get cost from ang vel         [error * penalty]
        # penaltyTetherUsage = tetherUsage * self.penaltyTetherUsage                  # get cost from tether force    [usage * penalty]
        penaltyDiffTetherUsage = deltaTetherUsage * self.penaltyDiffTetherUsage     # get cost from dtether force   [change in usage * penalty]

        # check if out of geofence
        outOfGeoFence = np.linalg.norm(obs[0:2]) > self.geoFenceMax_XY or obs[2] > self.geoFenceMax_Z or obs[2] < self.geoFenceMin_Z

        crashed = True if obs[2]<self.COLLISION_H else False                        # check if drone is crashed
        penaltyFlag = self.penaltyFlag if outOfGeoFence or crashed else 0           # set penaltyFlag if geoFence or crashed
        
        rewardPosition = np.sqrt(errorPosition)<self.positionThreshold                          # determine if position is within threshold
        rewardVelocity = np.sqrt(errorVelocity)<self.velocityThreshold                          # determine if lin vel is within threshold
        rewardAngularVelocity = np.sqrt(errorAngularVelocity)<self.angularVelocityThreshold     # determine if ang vel is within threshold
        
        if all([rewardPosition, rewardVelocity, rewardAngularVelocity]):            # if all conditions are met, add goal reward, else don't
            rewardGoal = self.rewardGoal
        else:
            rewardGoal = 0
            
        #actions = np.abs(actions)
        #penaltyControl = np.sum(np.square(actions-self.HOVER_RPM))*self.penaltyControl
        #penaltyDiffControl = np.square(np.max(actions)-np.min(actions))*self.penaltyDiffControl
        

        # sum and return reward
        return rewardGoal - penaltyPosition - penaltyAngle - penaltyVelocity - penaltyAngularVelocity \
               - penaltyFlag + penaltyDiffTetherUsage# - penaltyTetherUsage - penaltyControl - penaltyDiffControl

    ####################################################################################################
    #### Compute the current done value(s) #############################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - obs (..)                         the return of _computeObs() ################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - done (..)                        the done value(s) associated to the current obs/state ######
    ####################################################################################################
    def _computeDone(self, obs):
        outOfGeoFence = np.linalg.norm(obs[0:2]) > self.geoFenceMax_XY or \
                        obs[2] > self.geoFenceMax_Z or \
                        obs[2] < self.geoFenceMin_Z
        outOfTime = True if (self.step_counter/self.SIM_FREQ > 10) else False
        crashed = True if obs[2]<self.COLLISION_H else False
        return outOfGeoFence or outOfTime or crashed 

    ####################################################################################################
    #### Compute the current info dict(s) ##############################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - obs (..)                         the return of _computeObs() ################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - info (..)                        the info dict(s) associated to the current obs/state #######
    ####################################################################################################
    def _computeInfo(self, obs):
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ####################################################################################################
    #### Normalize the 20 values in the simulation state to the [-1,1] range ###########################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - state ((20,1) array)             raw simulation state #######################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - normalized state ((20,1) array)  clipped and normalized simulation state ####################
    ####################################################################################################
    def _clipAndNormalizeState(self, state):
        clipped_pos = np.clip(state[0:3], -1, 1)
        clipped_rp = np.clip(state[3:5], -np.pi/3, np.pi/3)
        clipped_vel = np.clip(state[6:9], -5, 5)
        clipped_ang_vel_rp = np.clip(state[9:11], -10*np.pi, 10*np.pi)
        clipped_ang_vel_y = np.clip(state[11], -20*np.pi, 20*np.pi)
        # if self.GUI: self._clipAndNormalizeStateWarning(state, clipped_pos, clipped_rp, clipped_vel, clipped_ang_vel_rp, clipped_ang_vel_y)
        normalized_pos = clipped_pos
        normalized_rp = clipped_rp/(np.pi/3)
        normalized_y = state[5]/np.pi
        normalized_vel = clipped_vel/5
        normalized_ang_vel_rp = clipped_ang_vel_rp/(10*np.pi)
        normalized_ang_vel_y = clipped_ang_vel_y/(20*np.pi)
        return np.hstack([normalized_pos, normalized_rp, normalized_y, normalized_vel, normalized_ang_vel_rp, normalized_ang_vel_y, state[12:16] ]).reshape(16,)

    ####################################################################################################
    #### Print a warning if any of the 20 values in a state vector is out of the normalization range ###
    ####################################################################################################
    def _clipAndNormalizeStateWarning(self, state, clipped_pos, clipped_rp, clipped_vel, clipped_ang_vel_rp, clipped_ang_vel_y):
        if not(clipped_pos==np.array(state[0:3])).all(): print("[WARNING] it", self.step_counter, "in RLTakeoffAviary._clipAndNormalizeState(), out-of-bound position [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLTakeoffAviary._computeDone()".format(state[0], state[1], state[2]))
        if not(clipped_rp==np.array(state[3:5])).all(): print("[WARNING] it", self.step_counter, "in RLTakeoffAviary._clipAndNormalizeState(), out-of-bound roll/pitch [{:.2f} {:.2f}], consider a more conservative implementation of RLTakeoffAviary._computeDone()".format(state[3], state[4]))
        if not(clipped_vel==np.array(state[6:9])).all(): print("[WARNING] it", self.step_counter, "in RLTakeoffAviary._clipAndNormalizeState(), out-of-bound velocity [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLTakeoffAviary._computeDone()".format(state[6], state[7], state[8]))
        if not(clipped_ang_vel_rp==np.array(state[9:11])).all(): print("[WARNING] it", self.step_counter, "in RLTakeoffAviary._clipAndNormalizeState(), out-of-bound angular velocity [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLTakeoffAviary._computeDone()".format(state[9], state[10], state[11]))
        if not(clipped_ang_vel_y==np.array(state[11])): print("[WARNING] it", self.step_counter, "in RLTakeoffAviary._clipAndNormalizeState(), out-of-bound angular velocity [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLTakeoffAviary._computeDone()".format(state[9], state[10], state[11]))

    ####################################################################################################
    #### Denormalize the [-1,1] range to the [0, MAX RPM] & [0, MAX_TETHER_FORCE] range ################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - action ((N_ACTIONS,1) array)     normalized [-1,1] actions applied to the 4 motors ##########
    ####################################################################################################
    #### Returns #######################################################################################
    #### - rpm ((N_ACTIONS,1) array)        RPM values to apply to the 4 motors ########################
    #### - tether_force_mag                 force applied by tether ####################################
    ####################################################################################################
    def _normalizedActionToRPM(self, action):
        if np.any(np.abs(action[0:self.N_ACTIONS-1]))>1:
            print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        rpm = np.where(action[0:self.N_ACTIONS-1] <= 0, (action[0:self.N_ACTIONS-1]+1)*self.HOVER_RPM, action[0:self.N_ACTIONS-1]*self.MAX_RPM) # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM
        tether = action[self.N_ACTIONS-1]*self.MAX_TETHER_FORCE # Non-linear mapping: 0 -> 0, 1 -> MAX_TETHER_FORCE
        return rpm, tether

    ####################################################################################################
    #### Return the state vector of the nth drone ######################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - nth_drone (int)                  order position of the drone in list self.DRONE_IDS #########
    ####################################################################################################
    #### Returns #######################################################################################
    #### - state ((21,) array)              the state vector of the nth drone ##########################
    ####################################################################################################
    def _getDroneStateVector(self, nth_drone):
        state = np.hstack([self.pos[nth_drone,:], self.quat[nth_drone,:], self.rpy[nth_drone,:], self.vel[nth_drone,:], self.ang_v[nth_drone,:], self.last_action[nth_drone,:]])
        return state.reshape(21,)        

    ####################################################################################################
    #### Set initial xyz for current episode ###########################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - initial_xyzs ((3,1) array)     initial x, y, and z position of drone in episode #############
    ####################################################################################################
    def _setInitialXYZ(self, initial_xyzs):
        return np.vstack([
                            random.uniform(-self.INIT_X_BOUND, self.INIT_X_BOUND),
                            random.uniform(-self.INIT_Y_BOUND, self.INIT_Y_BOUND),
                            random.uniform(-self.INIT_Z_BOUND, self.INIT_Z_BOUND)+1.0
                                                                ])

    ####################################################################################################
    #### Set initial rpy for current episode ###########################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - initial_rpys ((3,1) array)     initial r, p, and y orientations of drone in episode #########
    ####################################################################################################
    def _setInitialRPY(self, initial_rpys):
        return np.vstack([
                            random.uniform(-self.INIT_R_BOUND, self.INIT_R_BOUND),
                            random.uniform(-self.INIT_P_BOUND, self.INIT_P_BOUND),
                            random.uniform(-self.INIT_YAW_BOUND, self.INIT_YAW_BOUND)
                                                                ])
        
 
