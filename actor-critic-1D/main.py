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
NUM_EPISODES = 100000


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
    S, p_network, v_network, p_params, v_params = build_networks(num_actions=NUM_ACTIONS, state_length=STATE_LENGTH)
    R = tf.placeholder("float", [None], name="reward")
    A = tf.placeholder("float", [None, NUM_ACTIONS], name="action")

    # summary variable
    loss_sum_holder = tf.Variable(0.)

    # loss calculation (for one batch) operations
    log_prob = tf.log(tf.reduce_sum(p_network * A, reduction_indices=1))
    p_loss = -log_prob * (R - v_network)
    v_loss = tf.reduce_mean(tf.square(R - v_network)) / 2
    total_loss = p_loss + v_loss

    # optimization operations
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    minimize_op = optimizer.minimize(total_loss)

    R_mean = tf.reduce_mean(R)
    V_mean = tf.reduce_mean(v_network)
    tf.summary.scalar("Batch Reward", R_mean)
    tf.summary.scalar("Batch Value", V_mean)
    summary_op = tf.summary.merge_all()

    #return S, A, R, p_network, v_network, minimize_op, loss_sum_op, loss_clear_op, loss_sum_holder
    return S, A, R, p_network, v_network, minimize_op, summary_op


def run_agent(env, sess, graph_ops, writer):
    # graph_ops を unpack.
    #S, A, R, p_network, v_network, minimize_op, loss_sum_op, loss_clear_op, loss_sum_holder = graph_ops
    S, A, R, p_network, v_network, minimize_op, summary_op = graph_ops

    # summary variables.
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

            # state value 取得．
            val = sess.run(v_network, feed_dict={S: [s]})[0]
            episode_avg_value += val

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

        # train & summarize
        a_batch = np.array(a_batch)
        _, summary_str = sess.run([minimize_op, summary_op], feed_dict={R: r_sum_batch,
                                                                        A: a_batch,
                                                                        S: s_batch})

        writer.add_summary(summary_str, float(episode_number))

        if done:
            # Episode終了(=ゲームクリア)．データを統計してリセット
            # Episode終了していなければリセットせず，続きから．
            # episode reward はかかった時間に応じて変化したりしてくれるのか？
            if episode_number % 100 == 0:
                print("Episode:", episode_number, "| Time:", episode_time, "| Reward", episode_reward)

            episode_number += 1
            episode_time = 0
            episode_reward = 0
            done = False
            s = env.reset()


def train(sess, graph_ops):
    env = gym.make(ENV_NAME)

    writer = tf.summary.FileWriter(SUMMARY_PATH, sess.graph)
    sess.run(tf.global_variables_initializer())

    run_agent(env, sess, graph_ops, writer)


def setup_summaries(sess):
    # Not separated now.
    pass


def main():
    graph_ops = build_graph()
    with tf.Session() as sess:
        K.set_session(sess)
        train(sess, graph_ops)


main()
