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


class InceptionModel(object):
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

        return y_prob, f1_cost(y_prob, labels)
