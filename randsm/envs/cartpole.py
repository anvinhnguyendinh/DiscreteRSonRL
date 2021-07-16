"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import torch
import numpy as np


class CartPoleEnv(object):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        # high = np.array([self.x_threshold * 2,
        #                  np.finfo(np.float32).max,
        #                  self.theta_threshold_radians * 2,
        #                  np.finfo(np.float32).max],
        #                 dtype=np.float32)

        # self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # self.seed()
        # self.viewer = None
        # self.state = None

        # self.steps_beyond_done = None
        self.start_flag = True #
        self.done = None

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def __call__(self, state, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        x_ = state.select(1, 0)
        x_dot_ = state.select(1, 1)
        theta_ = state.select(1, 2)
        theta_dot_ = state.select(1, 3)

        if self.start_flag:
            done2 = torch.logical_or(x_ < -self.x_threshold, x_ > self.x_threshold)
            done1 = torch.logical_or(theta_ < -self.theta_threshold_radians, done2)
            done0 = torch.logical_or(theta_ >  self.theta_threshold_radians, done1)
            # self.start_flag  = False
        else:
            done0 = self.done

        force = self.force_mag * (2 * action - 1.0)
        costheta = torch.cos(theta_)
        sintheta = torch.sin(theta_)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot_ ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x_ + self.tau * x_dot_
            x_dot = x_dot_ + self.tau * xacc
            theta = theta_ + self.tau * theta_dot_
            theta_dot = theta_dot_ + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot_ + self.tau * xacc
            x = x_ + self.tau * x_dot_
            theta_dot = theta_dot_ + self.tau * thetaacc
            theta = theta_ + self.tau * theta_dot_

        _state = torch.stack((x, x_dot, theta, theta_dot), 1)

        done4 = torch.logical_or(x  < -self.x_threshold, x  > self.x_threshold)
        done3 = torch.logical_or(theta  < -self.theta_threshold_radians, done4)
        done  = torch.logical_or(theta  >  self.theta_threshold_radians, done3)

        # done = bool(
        #     x < -self.x_threshold
        #     or x > self.x_threshold
        #     or theta < -self.theta_threshold_radians
        #     or theta > self.theta_threshold_radians
        # )

        reward = 1.0 - torch.logical_and(done, done0).type(torch.float) # and / or

        self.done = done

        # if not done:
        #     reward = 1.0
        # elif self.steps_beyond_done is None:
        #     # Pole just fell!
        #     self.steps_beyond_done = 0
        #     reward = 1.0
        # else:
        #     # if self.steps_beyond_done == 0:
        #     #     logger.warn(
        #     #         "You are calling 'step()' even though this "
        #     #         "environment has already returned done = True. You "
        #     #         "should always call 'reset()' once you receive 'done = "
        #     #         "True' -- any further steps are undefined behavior."
        #     #     )
        #     self.steps_beyond_done += 1
        #     reward = 0.0

        return _state, reward, done