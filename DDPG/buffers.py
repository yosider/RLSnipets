# coding: utf-8
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, terminal, s_next):
        exp = (s, a, r, terminal, s_next)
        if self.count < self.buffer_size:
            self.buffer.append(exp)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(exp)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        """
        batch_size : １サンプルあたりのデータ数
        """
        batch = []

        if self.count < batch_size:
            # 足りない場合はあるだけ全てサンプル．
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([exp[0] for exp in batch])
        a_batch = np.array([exp[1] for exp in batch])
        r_batch = np.array([exp[2] for exp in batch])
        t_batch = np.array([exp[3] for exp in batch])
        s2_batch = np.array([exp[4] for exp in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
