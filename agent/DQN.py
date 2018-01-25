# coding: utf-8

import numpy as np

#from networks.CNN import build_encoder
from utils.Qnet import build_Qnet

class DQN_agent():
    def __init__(self, action_list):
        self.Q = build_Qnet

    def get_action(self, state):
        action_index = np.argmax(self.Q.predict(state))
        action = action_list[action]

    def update(self, state, reward):
