#!/usr/bin/env python
"""
File: probabalistic_model
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
import tensorflow as tf

from HumanProteinAtlas import Organelle
# from deep_model.ops import *

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


class ProbabilisticModel(object):
    def __init__(self, params):
        self.params = params

        self.variables_to_save = list()
        self.restore_tensors = list()

    def __call__(self, input, latent_priors, labels, is_training):
        assert isinstance(input, tf.Tensor)
        assert isinstance(labels, tf.Tensor)
        assert isinstance(latent_priors, tf.Tensor)
        assert isinstance(is_training, tf.Tensor)

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

        self.restore_tensors = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        def LatentConditional(input):
            with tf.variable_scope("Parallel"):
                parallel_layers = list()
                m, latent_size = latent_priors.shape
                for z in range(latent_size):
                    with tf.variable_scope("Track%d" % z):
                        l = batch_pool_drop_conv_relu(input, 128)

                        l = BatchNormalization(l)
                        l = MaxPooling2D(l)
                        l = tf.layers.dropout(l, rate=self.params.dropout_rate, training=is_training)

                        l = tf.layers.flatten(l)

                        with tf.variable_scope("FinalDense"):
                            l = tf.layers.dropout(l, rate=self.params.dropout_rate, training=is_training)
                            l = tf.layers.dense(l, len(Organelle), activation='relu')

                            l = BatchNormalization(l)
                            l = tf.layers.dropout(l, rate=self.params.dropout_rate, training=is_training)
                            logits = tf.layers.dense(l, len(Organelle), activation=None)

                            # Probability of class given input and latent variable `z`
                            py_xzi = tf.sigmoid(logits, name="py_xz%d" % z)
                            parallel_layers.append(py_xzi)

                # Conditional Probability of class, given X and latent variable(s)
                with tf.variable_scope("Py_xz"):
                    py_xz = tf.stack(parallel_layers, axis=2)
                return py_xz

        with tf.variable_scope("MarginalizeLatent"):
            pl_xc = LatentConditional(l)

            # Joint probability over class and latent
            with tf.variable_scope("Pyz_x"):
                pz_x = tf.keras.backend.repeat(latent_priors, len(Organelle))
                pyz_x = tf.multiply(pl_xc, pz_x, name="JointProbability")

            # Marginalize Latent
            py_x = tf.reduce_sum(pyz_x, axis=2)

        return py_x, f1_cost(py_x, labels)
