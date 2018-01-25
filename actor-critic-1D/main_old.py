# coding: utf-8

import numpy as np
import tensorflow as tf
from keras import backend as K
from model import build_networks
import gym

ENV_NAME = "CartPole-v0"
SUMMARY_PATH = "summaries/" + ENV_NAME
NUM_ACTIONS = 2
STATE_LENGTH = 4

BATCH_SIZE = 32

GAMMA = 0.99
LEARNING_RATE = 1e-5
NUM_EPISODES = 100


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

    #loss_summary_placeholder = summary_ops[4]
    #update_loss_summary = summary_ops[5]
    #summary_op = summary_ops[6]
    #writer = summary_ops[7]

    # 大文字：placeholder
    S, p_network, v_network, p_params, v_params = build_networks(num_actions=NUM_ACTIONS, state_length=STATE_LENGTH)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    R = tf.placeholder("float", [None], name="reward")
    A = tf.placeholder("float", [None, NUM_ACTIONS], name="action")
    # log_prob = log(Pr(action_index))
    log_prob = tf.log(tf.reduce_sum(p_network * A, reduction_indices=1))
    p_loss = -log_prob * (R - v_network)
    v_loss = tf.reduce_mean(tf.square(R - v_network)) / 2

    total_loss = p_loss + v_loss
    tf.summary.scalar('Loss', total_loss)

    minimize_op = optimizer.minimize(total_loss)
    return S, A, R, minimize_op, p_network, v_network


def run_agent(env, sess, graph_ops, summary_ops):
    episode_times = []
    episode_rewards = []

    # graph_ops を unpack.
    S, A, R, minimize_op, p_network, v_network = graph_ops

    # summary_ops を unpack.
    r_summary_placeholder = summary_ops[0]
    update_episode_reward = summary_ops[1]
    val_summary_placeholder = summary_ops[2]
    update_episode_value = summary_ops[3]
    summary_op = summary_ops[4]
    writer = summary_ops[5]

    episode_number = 0
    episode_time = 0
    episode_avg_value = 0
    episode_reward = 0

    done = False
    s = env.reset()

    while episode_number <= NUM_EPISODES:
        s_batch = []
        a_batch = []
        r_batch = []

        batch_length = 0

        while (not done) and (batch_length < BATCH_SIZE):
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

            batch_length += 1
            episode_time += 1
            s = s_next

        if done:
            r_sum = 0
        else:
            # Episodeが終わってなければ次のbatchは続きから．
            r_sum = sess.run(v_network, feed_dict={S: [s]})[0][0]

        r_sum_batch = np.zeros(batch_length)
        for i in reversed(range(batch_length)):
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

            # summarize
            sess.run(update_episode_reward, feed_dict={r_summary_placeholder: episode_reward})
            if episode_number % 10 == 0:
                summary_str = sess.run(summary_op)
                writer.add_summary(summary_str, float(episode_number))

            print("Episode:", episode_number, "| Time:", episode_time, "| Reward", episode_reward)
            episode_times.append(episode_time)
            episode_rewards.append(episode_reward)

            episode_number += 1
            episode_time = 0
            episode_reward = 0
            done = False
            s = env.reset()

    return episode_times, episode_rewards


import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
def visualize(x, y, name):
    plt.plot(x, y)
    plt.savefig(name)
    plt.clf()  # clear.


def train(sess, graph_ops, summary_ops):
    env = gym.make(ENV_NAME)

    sess.run(tf.global_variables_initializer())
    #writer = tf.summary.FileWriter(SUMMARY_PATH, sess.graph)

    run_agent(env, sess, graph_ops, summary_ops)


def setup_summaries(sess):
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Episode Reward", episode_reward)
    r_summary_placeholder = tf.placeholder("float")
    update_r_summary = episode_reward.assign(r_summary_placeholder)

    episode_avg_value = tf.Variable(0.)
    tf.summary.scalar("Episode Value", episode_avg_value)
    val_summary_placeholder = tf.placeholder("float")
    update_val_summary = episode_reward.assign(val_summary_placeholder)

    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(SUMMARY_PATH, sess.graph)

    return r_summary_placeholder, update_r_summary, val_summary_placeholder, update_val_summary, summary_op, writer


def main():
    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        K.set_session(sess)
        graph_ops = build_graph()
        summary_ops = setup_summaries(sess)

        train(sess, graph_ops, summary_ops)
        #visualize(np.arange(len(times)), times, "times_tmp.png")
        #visualize(np.arange(len(rewards)), rewards, "rewards_tmp.png")

main()
