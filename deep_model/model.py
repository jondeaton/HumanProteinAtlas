#!/usr/bin/env python
"""
File: model
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
import tensorflow as tf


from HumanProteinAtlas import Organelle
from deep_model.ops import *


class BaselineModel(object):

    def __init__(self, params):
        self.params = params

    def __call__(self, input, labels, is_training):

        layers = list()
        layers.append(input)

        with tf.variable_scope("convolutional"):
            num_conv = 2
            filter_count = [64, 32]
            for i in range(num_conv):
                with tf.variable_scope("layer-%d" % i):
                    next_layer = self._down_block(layers[-1], is_training, filter_count[i])
                    layers.append(next_layer)

        last_conv_layer = layers[-1]
        new_shape = np.prod(last_conv_layer.shape[1:])
        reshaped = tf.reshape(layers[-1], [-1,] + [new_shape])
        layers.append(reshaped)

        num_dense = 1
        with tf.variable_scope("dense"):
            sizes = [64]
            for i in range(num_dense):
                with tf.variable_scope("layer-%d" % i):
                    next_layer = tf.layers.dense(layers[-1], sizes[i], activation='relu')
                    layers.append(next_layer)

                    if self.params is not None and self.params.dropout:
                        dropout_layer = tf.layers.dropout(inputs=next_layer,
                                                          rate=self.params.dropout_rate,
                                                          training=is_training)
                        layers.append(dropout_layer)

        last_dense_layer = layers[-1]
        n_logits = len(Organelle)

        with tf.variable_scope("output"):
            logits = tf.layers.dense(last_dense_layer, n_logits, name="logits")
            output = tf.sigmoid(logits, name="probabilities")

        return output, logits, self._cost(logits, labels)

    def _cost(self, logits, labels):
        with tf.variable_scope("cost"):
            if self.params.cost == "unweighted":
                xS = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                             logits=logits)
            elif self.params.cost == 'f1':
                return f1_cost(tf.sigmoid(logits), labels)

            else:
                # weighted cross-entropy
                xS = tf.nn.weighted_cross_entropy_with_logits(targets=labels,
                                                              logits=logits,
                                                              pos_weight=self.params.positive_weight)

            cost = tf.reduce_mean(xS)
            return cost

    def _down_block(self, input, is_training, num_filters, name='down_level'):
        """ two convolutional blocks followed by a max-pooling layer

        :param input: 2D input tensor
        :param is_training: whether it is training or not
        :param num_filters: the number of filters ot use
        :param name: name of thie block
        :return: output of the max-pooling layer
        """
        with tf.variable_scope(name):
            conv1 = self._conv_block(input, is_training, num_filters=num_filters, name='conv1')
            # conv2 = self._conv_block(conv1, is_training, num_filters=num_filters, name='conv2')

            max_pool = tf.layers.max_pooling2d(conv1,
                                               pool_size=(2,2), strides=2,
                                               data_format='channels_first', name='max_pool')
            return max_pool

    def _conv_block(self, input, is_training, num_filters, name='conv'):
        """ Convolution and batch normalization layer

        :param input: The input tensor
        :param is_training: Boolean tensor whether it is being run on training or not
        :param num_filters: The number of filters to convolve on the input
        :param name: Name of the convolutional block
        :return: Tensor after convolution and batch normalization
        """
        with tf.variable_scope(name):
            kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
            bias_initializer = tf.zeros_initializer(dtype=tf.float32)

            conv = tf.layers.conv2d(input,
                                    filters=num_filters, kernel_size=(3,3), strides=(1,1), padding='same',
                                    data_format='channels_first', activation=None, use_bias=True,
                                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

            # Batch normalization before activation
            if self.params.batch_normalize:
                bn = tf.layers.batch_normalization(conv,
                                               axis=-1, momentum=0.9,
                                               epsilon=0.001, center=True, scale=True,
                                               training=is_training, name='bn')

                # Activation after batch normalization
                act = tf.nn.relu(bn, name="bn-relu")
            else:
                act = tf.nn.relu(conv, name="conv-relu")

            tf.summary.histogram('activations', act)
            tf.summary.scalar('sparsity', tf.nn.zero_fraction(act))
            return act


class InceptionBased(object):
    def __init__(self, params):
        self.params = params

        self.variables_to_save = list()

    def __call__(self, input, labels, is_training):

        def BatchNormalization(input):
            bn = tf.layers.batch_normalization(input, axis=-1, training=is_training)
            assert isinstance(bn, tf.keras.layers.BatchNormalization)

            self.variables_to_save.append(bn.moving_mean)
            self.variables_to_save.append(bn.moving_variance)

            return bn

        def ComplexBlock(input, filters):
            with tf.variable_scope("ComplexBlock"):
                x = BatchNormalization(input)
                x = MaxPooling2D(x)
                x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=is_training)
                return ConvReLu(x, filters, (3, 3))

        def BasicBlock(input, filters):
            with tf.variable_scope("BasicBlock"):
                intermediate = BatchNormalization(input)
                return ConvReLu(intermediate, filters, (3, 3))

        with tf.variable_scope("Basic"):
            l = BasicBlock(input, 8)
            l = BasicBlock(l, 8)
            l = BasicBlock(l, 16)

        with tf.variable_scope("Intermediate"):
            l = BatchNormalization(l)
            l = MaxPooling2D(l)

            l = tf.layers.dropout(l, rate=self.params.dropout_rate, training=is_training)

            l = inception_module(l, 16, [(3, 3), (5, 5), (7, 7), (1, 1)])
            l = BatchNormalization(l)

        with tf.variable_scope("Complex"):
            l = ComplexBlock(l, 32)
            l = ComplexBlock(l, 64)
            l = ComplexBlock(l, 128)

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
            y_prob = tf.sigmoid(logits)

        return y_prob, logits, f1_cost(y_prob, labels)
