#!/usr/bin/env python
"""
File: InceptionV1
Date: 12/7/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
import tensorflow as tf

from HumanProteinAtlas import Organelle
from deep_model.ops import *


class InceptionBased(object):
    def __init__(self, params):
        self.params = params

    def __call__(self, input, labels, is_training):

        def BatchNormalization(input):
            return tf.layers.batch_normalization(input, axis=-1, training=is_training)

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

        return y_prob, logits, f1_cost(y_prob, labels)
