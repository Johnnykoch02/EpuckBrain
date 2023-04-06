from gym import Env
from gym.spaces import MultiDiscrete, Box, Dict, Discrete
from multiprocessing.connection import Listener
import time
import threading as thr
import torch as th
from math import inf, radians, degrees
import numpy as np
import os
import time
import gzip
import re
from collections import deque
from enum import Enum
ROBOT_SERVER = ('localhost', 31313)     # family is deduced to be 'AF_INET'




class EpuckEnv(Env):
    '''This class is designed to implement control of the Berret Hand through use of SimController'''
    
    def _data_th(self,  ):
        conn = self.listener.accept()
        while True:
            try:
                data = conn.recv()
                # do something with msg
                # print('Message received: ', data)
                if(isinstance(data, str)):
                    if data == 'close':
                        conn.close()
                        break
                elif (isinstance(data, th.Tensor)):
                    print('Recieved data is a tensor')
                else: # Macro Type Updates
                    target = data['target']
                    if target == 'state':
                        self.obs.append(data['data'])
                        # print(self.obs[-1])
                        self.__Observe_Flag__ = True
                    elif target == 'start':
                        self.__Start_Flag__ = True
                    elif target == 'stop':
                        self.__Stop_Flag__ = True
            except EOFError as e:
                time.sleep(0.2)
                conn = self.listener.accept()
            
            time.sleep(0.05)
        self.listener.close()
    def __init__(self, connection=True):
        super().__init__()        

        spaces = {
            'delta_t': Box(low= -inf, high= inf, shape=(1, )), ### <Concatenation>
            'left_distance': Box(low= -inf, high= inf, shape=(1, )),
            'right_distance': Box(low= -inf, high= inf, shape=(1,)),
            'front_distance': Box(low= -inf, high= inf, shape=(1,)),
            'rear_distance': Box(low= -inf, high= inf, shape=(1,)),
            'previous_pos': Box(low= -inf, high= inf, shape=(1,)),
            'current_pos': Box(low = 0, high = 1, shape = (1, )), 
            'omega': Box(low= -inf, high= inf, shape=(1, )),
            'theta': Box(low= -inf, high= inf, shape=(1, )),
            'lidar': Box(low= -inf, high= inf, shape=(360,)),
            'current_cell': Box(low= -inf, high= inf, shape = (1,)),
            'cameraFront': Box(low= -inf, high= inf, shape=(3, 80, 80)),
            'cameraRear': Box(low= -inf, high= inf, shape=(3, 80, 80)),
            'cameraLeft': Box(low= -inf, high= inf, shape=(3, 80, 80)),
            'cameraRight' :Box(low= -inf, high= inf, shape=(3, 80, 80)),
            # 'previous_actions': Box(low= -inf, high= inf, shape=(15,)),
        }
        self.observation_space = Dict(spaces)
        self.obs = []
        
        #TODO Change this !!
        # self.action_space = MultiDiscrete([3, 3, 3])
        # self.action_space = MultiDiscrete([3, 3, 3, 3, 3, 3])
        self.action_space = Discrete(4)
        
        self.previous_actions = deque()
        for _ in range(80):
            self.previous_actions.append(np.zeros(shape=(3,)).astype(float))
        
        self.step_len = 0
        
        ### DATA FLAGS
        self.__Start_Flag__ = False
        self.__Stop_Flag__ = False
        self.__Observe_Flag__ = False
        
        
        #Reward Function Parameters
        self.termination_penalty = 150
        self.long_drop_penalty = 65
        self.success_reward = 175
        self.stabilizing_reward = 10
        self.failure_penalty = 120
        self.movement_penalty_scale_factor = 0.2
        
        if connection:
            self.listener = Listener(ROBOT_SERVER)
            self.data_th = thr.Thread(target=self._data_th)
            self.data_th.start()
        
    def perform_action(self, action):
        pass
    
    def get_state(self,):
        try:
            return self.obs[-1]
        except:
            return self.observation_space.sample()
        
    def step(self, action):
        print('Step Len:', self.step_len)
            # self.play_audio()
        while not self.__Observe_Flag__:
            time.sleep(0.1)
        self.__Observe_Flag__ = False
        self.step_len+=1

        # print('Step:', action)     
        
        state = self.get_state()
        reward, done = self.get_current_reward()
        self.perform_action(action)
        # Keep track of action encoding
        # self.previous_actions.append(self.one_hot_encode(action))
        # self.previous_actions.popleft()
        
        return state, reward, done, {}
    
    def reset(self):
        self.__Stop_Flag__ = False
        self.__Observe_Flag__ = False        
        while not self.__Start_Flag__:
            print('...Waiting for start...')
            time.sleep(0.5)
            
        self.__Start_Flag__ = False
        
        self.step_len = 0
    
    def get_current_reward(self):
        reward = 0
        done = False
        if self.__Stop_Flag__:
            done = True
        
        return reward, done
        