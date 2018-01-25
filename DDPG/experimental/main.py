# coding: utf-8
"""
DDPG
http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
"""

import numpy as np
import gym
import tensorflow as tf

from networks import ActorNetwork, CriticNetwork
from buffers import ReplayBuffer
from utils import OrnsteinUhlenbeckActionNoise

ENV = gym.make('Pendulum-v0')

STATE_DIM = ENV.observation_space.shape[0]
ACTION_DIM = ENV.action_space.shape[0]
ACTION_BOUND = ENV.action_space.high
# Ensure action bound is symmetric
assert (ENV.action_space.high == -ENV.action_space.low)

TAU = 0.001 # mixing rate for target network update
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
BUFFER_SIZE = 1000000
MINIBATCH_SIZE = 64
MAX_EPISODES = 500
MAX_EP_STEPS = 1000
GAMMA = 0.99

with tf.Session() as sess:

    critic = CriticNetwork(sess, STATE_DIM, ACTION_DIM, TAU, CRITIC_LEARNING_RATE)
    actor = ActorNetwork(sess, STATE_DIM, ACTION_DIM, ACTION_BOUND, MINIBATCH_SIZE, TAU, ACTOR_LEARNING_RATE, critic.model)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(ACTION_DIM))

    #TODO: Ornstein-Uhlenbeck noise.

    sess.run(tf.global_variables_initializer())

    # initialize target net
    actor.update_target_network()
    critic.update_target_network()

    # initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    # log variables
    reward_log = []

    # main loop.
    for ep in range(MAX_EPISODES):

        episode_reward = 0

        s = ENV.reset()

        for step in range(MAX_EP_STEPS):

            a = actor.predict(np.reshape(s, (1, STATE_DIM))) + actor_noise()
            s2, r, terminal, info = ENV.step(a[0])

            replay_buffer.add(np.reshape(s, (STATE_DIM,)), \
                              np.reshape(a, (ACTION_DIM,)), \
                              r, \
                              terminal, \
                              np.reshape(s2, (STATE_DIM,)))

            # Batch sampling.
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # target Q値を計算．
                target_action = actor.predict_target(s2_batch)
                target_q = critic.predict_target(s2_batch, target_action)

                # critic の target V値を計算．
                targets = []
                for i in range(MINIBATCH_SIZE):
                    if t_batch[i]:
                        # terminal
                        targets.append(r_batch[i])
                    else:
                        targets.append(r_batch[i] + GAMMA * target_q[i])

                # Critic を train.
                #TODO: predQはepisodeではなくrandom batchなのでepisode_avg_maxという統計は不適切．
                pred_q, _ = critic.train(s_batch, a_batch, np.reshape(targets, (MINIBATCH_SIZE, 1)))

                # Actor を　train.
                actor.train(s_batch, a_batch)

                # Update target networks.
                # 数batchに一度にするべき？
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            episode_reward += r
            episode_avg_max_Q += np.amax(pred_q)

            if terminal:
                print('Episode:', ep, 'Reward:', episode_reward)
                reward_log.append(episode_reward)
                Q_log.append(episode_avg_max_Q / step)

                break


import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.plot(reward_log)
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig('reward.png')
