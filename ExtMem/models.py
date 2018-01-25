import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model

def build_q_network(S, num_actions, state_length=2):
    inputs = Input(shape=(state_length,))
    dense1 = Dense(16, activation='relu', name='dense1')(inputs)
    dense2 = Dense(16, activation='relu', name='dense2')(dense1)

    q_vals = Dense(num_actions, activation='softmax', name='Q vals')(dense2)

    q_net = Model(inputs=inputs, outputs=q_vals)

    return q_net
    
