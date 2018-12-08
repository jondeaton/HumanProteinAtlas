#!/usr/bin/env python
"""
File: ops
Date: 11/21/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf


def f1(y_true, y_pred):
    with tf.variable_scope("macro-f1-score"):
        y_pred = tf.cast(y_pred, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        p = tp / (tp + fp + tf.keras.backend.epsilon())
        r = tp / (tp + fn + tf.keras.backend.epsilon())

        f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return tf.reduce_mean(f1)


def f1_cost(y_prob, y_true):
    with tf.variable_scope("macro-f1-loss"):
        tp = tf.reduce_sum(y_true * y_prob, axis=0)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_prob), axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_prob, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_prob), axis=0)

        p = tp / (tp + fp + tf.keras.backend.epsilon())
        r = tp / (tp + fn + tf.keras.backend.epsilon())

        f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return 1 - tf.reduce_mean(f1)


def inception_module(input, filters, kernels):
    with tf.variable_scope("Inception"):
        l = input
        convs = list()
        for kernel in kernels:
            l = ConvReLu(l, filters, kernel)
            convs.append(l)
        return tf.concat(values=convs, axis=1, name="concat")


def ConvReLu(input, filters, kernel):
    with tf.variable_scope("ConvReLu"):
        kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
        bias_initializer = tf.zeros_initializer(dtype=tf.float32)

        return tf.layers.conv2d(input,
                         filters=filters, kernel_size=kernel, strides=(1, 1), padding='same',
                         data_format='channels_first', activation='relu', use_bias=True,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)


def MaxPooling2D(x):
    return tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=2, data_format='channels_first')
