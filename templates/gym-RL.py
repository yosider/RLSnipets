# coding: utf-8

import numpy as np
import gym

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
