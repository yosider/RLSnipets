import gym
import numpy as np

from keras.models import Model
from keras.layers import Input, concatenate, Dense, Flatten, Activation, Reshape
from keras import backend as K

from matplotlib import pyplot as plt


def main():
    env = gym.make('Pendulum-v0')
    assert env.action_space.shape == (1,)
    assert env.observation_space.shape == (3,)

    agent = Agent(env.action_space.shape, env.observation_space.shape)

    for i in range(3000):
        # initial state
        state = env.reset()
        # Episode finished or not
        done = False

        for _ in range(200):  # one episode
            action = agent.get_next_action(state)
            state, reward, done, info = env.step(action)
            agent.store(state, action, reward, done, info)
            if done:
                break

        history = agent.train()

        plt.plot(history)
    return history



class Agent(object):
    def __init__(self, action_shape, state_shape):
        assert len(action_shape) == 1
        assert len(state_shape) == 1
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.predictor = self.build()
        self.action_memory = []
        self.state_memory = []

    def build(self):
        action_input = Input(shape=self.action_shape, name='action_input')
        state_input = Input(shape=self.state_shape, name='state_input')
        x = concatenate([action_input, state_input])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(self.state_shape[0])(x)
        #x = Reshape(self.state_shape)(x)
        predictor = Model(inputs=[action_input, state_input], outputs=[x])
        predictor.compile(optimizer='rmsprop',
                          loss='mse',
                          metrics=['accuracy'])
        return predictor

    def get_next_action(self, state):
        action = np.random.rand(1) *4 -2
        return action

    def store(self, state, action, reward, done, info):
        self.action_memory.append(action)
        self.state_memory.append(state)

    def train(self):
        X_action = np.array(self.action_memory[:-1])
        X_state = np.array(self.state_memory[:-1])
        Y = np.array(self.state_memory[1:])
        history = self.predictor.fit([X_action, X_state], [Y], epochs=100, batch_size=32)
        return history


main()
