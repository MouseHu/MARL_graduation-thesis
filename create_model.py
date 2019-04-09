import tensorflow as tf
import numpy as np


def dqn_model(observation, num_actions, num=0):
    convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]

    with tf.variable_scope('DeepQNetwork{}'.format(num)):
        out = observation
        for num_filters, kernel_size, stride in convs:
            out = tf.layers.conv2d(out, num_filters, kernel_size, strides=(stride, stride), activation=tf.nn.relu)
            #keras.layers.conv2d
        out = tf.layers.flatten(out)
        out = tf.nn.relu(tf.layers.dense(out, units=512))
        out = tf.layers.dense(out, units=num_actions)
    return out


def dist_dqn_model(observation, num_actions, num_atoms, num=0):
    convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]

    with tf.variable_scope('DistributionalDeepQNetwork'.format(num)):
        out = observation
        for num_filters, kernel_size, stride in convs:
            out = tf.layers.conv2d(out, num_filters, kernel_size, strides=(stride, stride), activation=tf.nn.relu)
        out = tf.layers.flatten(out)
        out = tf.nn.relu(tf.layers.dense(out, units=512))
        out = tf.layers.dense(out, units=num_actions * num_atoms)
        out = tf.reshape(out, shape=[-1, num_actions, num_atoms])
        out = tf.nn.softmax(out, dim=-1, name='softmax')
    return out
