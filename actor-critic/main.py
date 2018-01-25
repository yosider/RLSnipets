# coding: utf-8

import numpy as np
import tensorflow as tf
from keras import backend as K
from model import build_networks
import gym
from atari_environment import AtariEnvironment

ENV_NAME = "Breakout-v0"
NUM_ACTIONS = 3
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84

SEQUENCE_LENGTH = 4
BATCH_SIZE = 32

GAMMA = 0.99
LEARNING_RATE = 1e-5
NUM_EPISODES = 300


def sample_policy_action(num_actions, probs):
    """
    policy network が出力した確率分布に従って
    action を取得する．
    """
    # probs = [p(0), p(1), ..., p(num_actions)]
    # ただし p0 + p1 + ... = 1 をみたす

    # 確率の総和がたまに1を超えてエラーが出るので微小量を引いておく
    probs = probs - np.finfo(np.float32).epsneg

    # probsに従って1回サンプリングする．
    histogram = np.random.multinomial(1, probs)
    action_index = int(np.nonzero(histogram)[0])
    return action_index


def build_graph():
    # 大文字：placeholder
    S, p_network, v_network, p_params, v_params = build_networks(num_actions=NUM_ACTIONS, sequence_length=SEQUENCE_LENGTH, width=RESIZED_WIDTH, height=RESIZED_HEIGHT)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    R = tf.placeholder("float", [None], name="reward")
    A = tf.placeholder("float", [None, NUM_ACTIONS], name="action")
    # log_prob = log(Pr(action_index))
    log_prob = tf.log(tf.reduce_sum(p_network * A, reduction_indices=1))
    p_loss = -log_prob * (R - v_network)
    v_loss = tf.reduce_mean(tf.square(R - v_network)) / 2

    total_loss = p_loss + v_loss

    minimize_op = optimizer.minimize(total_loss)
    return S, A, R, minimize_op, p_network, v_network


def train(env, sess, graph_ops):
    episode_times = []
    episode_rewards = []

    # graph_ops を unpack.
    S, A, R, minimize_op, p_network, v_network = graph_ops

    batch_number = 0

    episode_number = 0
    episode_time = 0
    episode_reward = 0

    done = False
    s = env.get_initial_state()

    while episode_number <= NUM_EPISODES:
        s_batch = []
        a_batch = []
        r_batch = []

        batch_number += 1 # 今のところ未使用
        batch_time = 0

        while (not done) and (batch_time < BATCH_SIZE):
            # action取得．
            probs = sess.run(p_network, feed_dict={S: [s]})[0]
            action_index = sample_policy_action(NUM_ACTIONS, probs)
            a = np.zeros([NUM_ACTIONS])
            a[action_index] = 1

            # memoryに記憶．
            s_batch.append(s)
            a_batch.append(a)

            # envを更新，rewardの取得・記憶．
            s_next, r, done, info = env.step(action_index)
            episode_reward += r

            r = np.clip(r, -1, 1)
            r_batch.append(r)

            batch_time += 1
            episode_time += 1
            s = s_next

        if done:
            r_sum = 0
        else:
            # Episodeが終わってなければ次のbatchは続きから．
            r_sum = sess.run(v_network, feed_dict={S: [s]})[0][0]

        r_sum_batch = np.zeros(batch_time)
        for i in reversed(range(batch_time)):
            r_sum = r_batch[i] + GAMMA * r_sum
            r_sum_batch[i] = r_sum

        # train
        a_batch = np.array(a_batch)
        #print("r:", r_sum_batch.shape)
        #print("a:", a_batch.shape)
        sess.run(minimize_op, feed_dict={R: r_sum_batch,
                                         A: a_batch,
                                         S: s_batch})

        if done:
            # Episode終了(=ゲームクリア)．データを統計してリセット
            # Episode終了していなければリセットせず，続きから．
            # episode reward はかかった時間に応じて変化したりしてくれるのか？
            print("Episode:", episode_number, "| Time:", episode_time, "| Reward", episode_reward)
            episode_times.append(episode_time)
            episode_rewards.append(episode_reward)

            episode_number += 1
            episode_time = 0
            episode_reward = 0
            done = False
            s = env.get_initial_state()

    return episode_times, episode_rewards


import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
def visualize(x, y, name):
    plt.plot(x, y)
    plt.savefig(name)
    plt.clf()  # clear.


def main():
    env = gym.make(ENV_NAME)
    env = AtariEnvironment(gym_env=env, resized_width=RESIZED_WIDTH, resized_height=RESIZED_HEIGHT, sequence_length=SEQUENCE_LENGTH)

    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        K.set_session(sess)
        graph_ops = build_graph()
        sess.run(tf.global_variables_initializer())

        times, rewards = train(env, sess, graph_ops)
        print(times)
        print(rewards)
        visualize(np.arange(len(times)), times, "times.png")
        visualize(np.arange(len(rewards)), rewards, "rewards.png")

main()
