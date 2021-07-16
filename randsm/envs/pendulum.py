# Copied from OpenAI Gym

import torch
import numpy as np


class PendulumEnv(object):
    # metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second': 30
    # }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # high = np.array([1., 1., self.max_speed], dtype=np.float32)
        # self.action_space = spaces.Box(
        #     low=-self.max_torque,
        #     high=self.max_torque, shape=(1,),
        #     dtype=np.float32
        # )
        # self.observation_space = spaces.Box(
        #     low=-high,
        #     high=high,
        #     dtype=np.float32
        # )
        # self.seed()
        self.clip0 = torch.FloatTensor([ 1.0,  1.0,  8.0]).to(self.device)
        self.clip1 = torch.FloatTensor([-1.0, -1.0, -8.0]).to(self.device)
        self.mean = torch.FloatTensor([0.75143003, 0.00285801, 0.0622901]).to(self.device) # device a2c
        self.std  = torch.sqrt(torch.FloatTensor([0.31303893, 0.12230581, 3.50250079]) + 1e-8).to(self.device) # * self.std + self.mean

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def __call__(self, state, action):
        # th, thdot = self.state  # th := theta
        state_ = torch.max(torch.min(state, self.clip0), self.clip1)

        cth = state_.select(1, 0)
        sth = state_.select(1, 1)
        thdot = state_.select(1, 2)

        th = torch.acos(cth) * torch.sign(sth)

        # g = self.g
        # m = self.m
        # l = self.l
        # dt = self.dt

        u = torch.clip(action, -self.max_torque, self.max_torque)
        # self.last_u = u  # for rendering
        costs = th ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (3 * self.g / (2 * self.l) * sth + 3. / (self.m * self.l ** 2) * u) * self.dt
        newth = th + newthdot * self.dt
        newthdot = torch.clip(newthdot, -self.max_speed, self.max_speed)

        # self.state = np.array([newth, newthdot])
        _state = torch.stack((torch.cos(newth), torch.sin(newth), newthdot), 1)
        # _state = (_state - self.mean) / self.std
        return _state, -costs, False