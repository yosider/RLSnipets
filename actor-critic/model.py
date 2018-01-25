import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.models import Model

def build_networks(num_actions, sequence_length, width, height):
    with tf.device("/cpu:0"):
        S = tf.placeholder("float", [None, sequence_length, width, height], name="state")

        inputs = Input(shape=(sequence_length, width, height,))
        shared = Conv2D(16, (8,8), strides=4, activation='relu', padding='same', name="conv1")(inputs)
        shared = Conv2D(32, (4,4), strides=2, activation='relu', padding='same', name="conv2")(shared)
        shared = Flatten()(shared)
        shared = Dense(256, activation='relu', name="h1")(shared)

        action_probs = Dense(num_actions, activation='softmax', name="p")(shared)

        state_value = Dense(1, activation='linear', name="v")(shared)

        policy_network = Model(inputs=inputs, outputs=action_probs)
        value_network = Model(inputs=inputs, outputs=state_value)

        p_params = policy_network.trainable_weights
        v_params = value_network.trainable_weights

        p_out = policy_network(S)
        v_out = value_network(S)

    return S, p_out, v_out, p_params, v_params
