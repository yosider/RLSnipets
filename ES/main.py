# coding: utf-8
"""
Evolution Strategies

Evolution Strategies as a Scalable Alternative to Reinforcement Learning
(Salimans et al., Sep, 2017)
https://arxiv.org/abs/1703.03864

https://github.com/MorvanZhou/Evolutionary-Algorithm/blob/master/tutorial-contents/Using%20Neural%20Nets/Evolution%20Strategy%20with%20Neural%20Nets.py
http://mabonki0725.hatenablog.com/entry/2017/09/02/194737
"""

import numpy as np
import gym

LEARNING_RATE = 0.001
NOISE_VAR = 1.0


ENV = gym.make('Pendulum-v0')
STATE_DIM = ENV.observation_space.shape[0]
ACTION_DIM = ENV.action_space.shape[0]
ACTION_BOUND = ENV.action_space.high[0]

MAX_EPISODES = 1000
MAX_EP_STEPS = 1000

for ep in range(MAX_EPISODES):
    s = ENV.reset()

    for step in range(MAX_EP_STEPS):
        a = None
        s_next, r, done, info = ENV.step(a)

        if done:
            break
