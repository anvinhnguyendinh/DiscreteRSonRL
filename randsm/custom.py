import numpy as np
from .envs import *

class Custom(object):
    def __init__(self, classes, scaler, clip, num, reward_cl, torch_environ, smooth_dict, action_space, reward_space):
        self.CLAS = classes
        self.SCAL = scaler
        self.LIMI = clip
        self.FRAM = num
        self.RECL = reward_cl
        self.ACSP = action_space
        self.RESP = reward_space
        self.TENV = torch_environ

        self.DICT = smooth_dict

    def eval(self, algo):
        self.SCAL = self.SCAL[algo]
        self.TENV = self.TENV()


ENV_CUSTOM_INFO = {'CartPole-v1': Custom(2, {'ppo': np.array([4.8, 1.542 , 0.418, 1.842 ]), 'a2c': np.array([4.8, 2.080 , 0.418, 2.482 ])}, \
                                            np.array([2.4, np.inf, 0.209, np.inf]), 500, 2, CartPoleEnv, \
                                            {2: [14, 15], 1: [11, 15], 0: [9, 3]}, ( 0, 1, np.int  ),  (    0, 1, np.float)), \
                   'Pendulum-v0': Custom(0, {'ppo': np.array([2.0 , 2.0 , 16.0]), 'a2c': np.array([8.62, 8.62, 8.62])}, \
                                            np.array([1.0 , 1.0 ,  8.0]), 200, 0, PendulumEnv, \
                                            {2: [12, 15], 1: [ 9, 15], 0: [7, 3]}, (-2, 2, np.float),  (-16.3, 0, np.float)), \
                   }