# coding: utf-8

import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Lambda, BatchNormalization
from keras.layers.merge import concatenate
from keras.initializers import uniform
from keras.optimizers import Adam
import keras.backend as K

K.set_learning_phase(1)


class ActorNetwork(object):
    """
    input: state
    output: action
    """
    def __init__(self, sess, state_size, action_size, action_bound, batch_size, mixing_rate, learning_rate, critic_model):
        self.sess = sess
        self.tau = mixing_rate
        self.learning_rate = learning_rate

        K.set_session(sess)

        self.model, self.params, self.state = self.create_actor_network(state_size, action_size, action_bound)
        self.target_model, self.target_params, self.target_state = self.create_actor_network(state_size, action_size, action_bound)

        self.inputs = critic_model.inputs
        print("inputs", "\n", self.inputs)
        print("params", "\n", self.params)
        print("loss", "\n", -critic_model(self.inputs))
        update = self.model.optimizer.get_updates(
            params=self.params,
            loss=-critic_model(self.inputs))
        update += self.model.updates # for BN?
        self.optimize = K.function(self.inputs + [K.learning_phase()],
                                    [self.model(self.inputs[0])], update=update)

    def train(self, state, action):
        action = self.optimize([state, action])

    def train_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1-self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def predict(self, state):
        return self.model.predict(state)

    def predict_target(self, state):
        return self.target_model.predict(state)

    def create_actor_network(self, state_size, action_size, action_bound):
        state = Input(shape=[state_size])  # placeholder
        h = Dense(64, kernel_initializer='he_uniform', activation='relu')(state)
        h = BatchNormalization()(h)
        h = Dense(32, kernel_initializer='he_uniform', activation='relu')(h)
        action = Dense(action_size, kernel_initializer='random_uniform', activation='tanh')(h)
        action = Lambda(lambda x: x * action_bound)(action)
        model = Model(inputs=state, outputs=action)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)  # pseudo loss

        return model, model.trainable_weights, state




class CriticNetwork(object):
    """
    input: state, action
    output: Q(state, action)
    """
    def __init__(self, sess, state_size, action_size, mixing_rate, learning_rate):
        self.sess = sess
        self.tau = mixing_rate
        self.learning_rate = learning_rate

        K.set_session(sess)

        self.model, self.state, self.action = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_state, self.target_action = self.create_critic_network(state_size, action_size)

    def train(self, state, action, target_q):
        # TODO: can return loss value.
        self.model.train_on_batch([state, action], target_q)

    def train_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1-self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def predict(self, state, action):
        return self.model.predict([state, action])

    def predict_target(self, state, action):
        return self.target_model.predict([state, action])

    def create_critic_network(self, state_size, action_size):
        state = Input(shape=[state_size])  # placeholder
        action = Input(shape=[action_size])
        h = Dense(64, kernel_initializer='he_uniform', activation='relu')(state)
        h = BatchNormalization()(h)
        h = concatenate([h, action])
        h = Dense(32, kernel_initializer='he_uniform', activation='relu')(h)
        qval = Dense(action_size, kernel_initializer='random_uniform', activation='linear')(h)
        model = Model(inputs=[state,action], outputs=qval)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model, state, action
