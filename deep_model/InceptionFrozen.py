#!/usr/bin/env python
"""
File: InceptionFrozen
Date: 12/8/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""


import tensorflow as tf
from HumanProteinAtlas import Organelle

from deep_model.ops import f1_cost


def inception_module(input, filters, kernels):
    with tf.variable_scope("inception"):
        l = input
        convs = list()
        for kernel in kernels:
            l = conv_relu(l, filters, kernel)
            convs.append(l)
        return tf.concat(values=convs, axis=1, name="concat")


def conv_relu(input, filters, kernel):
    kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
    bias_initializer = tf.zeros_initializer(dtype=tf.float32)
    l = tf.layers.conv2d(input,
                         filters=filters, kernel_size=kernel, strides=(1, 1), padding='same',
                         data_format='channels_first', activation=None, use_bias=True,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    return tf.nn.relu(l)


def MaxPooling2D(x):
    return tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=2, data_format='channels_first')


class InceptionFrozen(object):
    def __init__(self, params):
        self.params = params
        self.variables_to_save = list()

    def __call__(self, input, labels, is_training):

        def BatchNormalization(input):
            batchnorm = tf.keras.layers.BatchNormalization(axis=-1)
            assert isinstance(batchnorm, tf.keras.layers.BatchNormalization)
            bn = batchnorm(input, training=is_training)

            # Need to add these to the list of variables to save
            # these attributes only become available after calling apply
            self.variables_to_save.append(batchnorm.moving_mean)
            self.variables_to_save.append(batchnorm.moving_variance)

            return bn

        def batch_conv_relu(input, filters):
            x = BatchNormalization(input)
            return conv_relu(x, filters, (3, 3))

        def batch_pool_drop_conv_relu(input, filters):
            x = BatchNormalization(input)
            x = MaxPooling2D(x)
            x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=is_training)
            return conv_relu(x, filters, (3, 3))

        l = batch_conv_relu(input, 8)
        l = batch_conv_relu(l, 8)
        l = batch_conv_relu(l, 16)

        l = BatchNormalization(l)
        l = MaxPooling2D(l)

        l = tf.layers.dropout(l, rate=self.params.dropout_rate, training=is_training)

        l = inception_module(l, 16, [(3, 3), (5, 5), (7, 7), (1, 1)])
        l = BatchNormalization(l)

        l = batch_pool_drop_conv_relu(l, 32)
        l = batch_pool_drop_conv_relu(l, 64)

        with tf.variable_scope("Trainable"):
            l = batch_pool_drop_conv_relu(l, 128)

            l = BatchNormalization(l)
            l = MaxPooling2D(l)
            l = tf.layers.dropout(l, rate=self.params.dropout_rate, training=is_training)

            l = tf.layers.flatten(l)
            l = tf.layers.dropout(l, rate=self.params.dropout_rate, training=is_training)
            l = tf.layers.dense(l, len(Organelle), activation='relu')

            l = BatchNormalization(l)
            l = tf.layers.dropout(l, rate=self.params.dropout_rate, training=is_training)
            logits = tf.layers.dense(l, len(Organelle), activation=None)
            y_prob = tf.sigmoid(logits)

        return y_prob, f1_cost(y_prob, labels)
