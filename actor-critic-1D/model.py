import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Dense, Input
from keras.models import Model

def build_networks(num_actions, state_length):
    with tf.device("/cpu:0"):
        S = tf.placeholder("float", [None, state_length], name="state")

        inputs = Input(shape=(state_length,))
        shared = Dense(16, activation='relu', name='dense1')(inputs)
        shared = Dense(16, activation='relu', name="dense2")(shared)

        action_probs = Dense(num_actions, activation='softmax', name="p")(shared)

        state_value = Dense(1, activation='linear', name="v")(shared)

        policy_network = Model(inputs=inputs, outputs=action_probs)
        value_network = Model(inputs=inputs, outputs=state_value)

        p_params = policy_network.trainable_weights
        v_params = value_network.trainable_weights

        p_out = policy_network(S)
        v_out = value_network(S)

    return S, p_out, v_out, p_params, v_params
