# coding: utf-8

import numpy as np

import tensorflow as tf
import tflearn
# kerasだとtf.gradientsがkeras.layers.BatchNormalizationをうまく微分してくれない(Noneになる．)

from pprint import pprint


class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, mixing_rate, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound  # scalar, action値の最大値．
        self.learning_rate = learning_rate
        self.tau = mixing_rate
        self.batch_size = batch_size

        # actor network
        self.state, self.out = self.create_actor_network()
        self.network_params = tf.trainable_variables()
        # target(fixed) actor network
        self.target_state, self.target_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        #pprint(self.network_params)
        #print('A_target_network_params', self.target_network_params)

        self.build_ops()

    def create_actor_network(self):
        state = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(state, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        out = tf.multiply(out, self.action_bound)
        return state, out

    def build_ops(self):
        # target network を更新するop
        self.update_target_network_params = \
            [self.target_network_params[i].assign( \
                tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1.-self.tau) \
            ) for i in range(len(self.target_network_params))]

        # critic により与えられる dQ/da
        self.action_gradients = tf.placeholder(tf.float32, [None, self.a_dim])

        # 各パラメータθiに対する dQ/dθi = dQ/da * da/dθi のリスト．
        # tf.gradient(y,x,c): c*dy/dx (たぶん．)
        unnormalized_actor_gradients = tf.gradients(self.out, self.network_params, -self.action_gradients)
        #pprint(self.unnormalized_actor_gradients)
        self.actor_gradients = [g/float(self.batch_size) for g in unnormalized_actor_gradients]
        #self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op.
        # 各パラメータに対して勾配を適用．
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))


    def train(self, state, action_gradients):
        """ run minimization."""
        self.sess.run(self.optimize, feed_dict={
            self.state: state,
            self.action_gradients: action_gradients
        })

    def predict(self, state):
        """ run action prediction and return the action."""
        return self.sess.run(self.out, feed_dict={
            self.state: state
        })

    def predict_target(self, state):
        """ run action prediction and return the action with TARGET network."""
        return self.sess.run(self.target_out, feed_dict={
            self.target_state: state
        })

    def update_target_network(self):
        """ run target network update op."""
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return len(self.network_params) + len(self.target_network_params)



class CriticNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, mixing_rate, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = mixing_rate

        # critic network
        self.state, self.action, self.out = self.create_critic_network()
        self.network_params = tf.trainable_variables()[num_actor_vars:]
        # target(fixed) critic network
        self.target_state, self.target_action, self.target_out = self.create_critic_network()
        self.target_network_params = tf.trainable_variables()[(num_actor_vars + len(self.network_params)):]

        #print('C_network_params', self.network_params)
        #print('C_target_network_params', self.target_network_params)

        self.build_ops()


    def create_critic_network(self):
        state = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(state, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return state, action, out


    def build_ops(self):
        # target network を更新するop
        self.update_target_network_params = \
            [self.target_network_params[i].assign( \
                tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1-self.tau) \
            ) for i in range(len(self.target_network_params))]

        # target Q value
        # target networkから取得．
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # loss, optimization op の定義．
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # d(Loss)/da
        self.action_grads = tf.gradients(self.out, self.action)


    def train(self, state, action, pred_q):
        """ run minimization."""
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.state: state,
            self.action: action,
            self.predicted_q_value: pred_q
        })

    def predict(self, state, action):
        """ run action prediction and return the action."""
        return self.sess.run(self.out, feed_dict={
            self.state: state,
            self.action: action
        })

    def predict_target(self, state, action):
        """ run action prediction and return the action with TARGET network."""
        return self.sess.run(self.target_out, feed_dict={
            self.target_state: state,
            self.target_action: action
        })

    def action_gradients(self, state, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: state,
            self.action: action
        })

    def update_target_network(self):
        """ run target network update op."""
        self.sess.run(self.update_target_network_params)
