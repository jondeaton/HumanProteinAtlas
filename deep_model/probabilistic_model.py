#!/usr/bin/env python
"""
File: probabalistic_model
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
import tensorflow as tf

from HumanProteinAtlas import Organelle
from deep_model.ops import *


class ProbabilisticModel(object):
    def __init__(self, params):
        self.params = params

        self.variables_to_save = list()

    def __call__(self, input, labels, latent_priors, is_training):
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

        def BasicBlock(input, filters):
            x = BatchNormalization(input)
            return ConvReLu(x, filters, (3, 3))

        def ComplexBlock(input, filters):
            x = BatchNormalization(input)
            x = MaxPooling2D(x)
            x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=is_training)
            return ConvReLu(x, filters, (3, 3))

        l = BasicBlock(input, 8)
        l = BasicBlock(l, 8)
        l = BasicBlock(l, 16)

        l = BatchNormalization(l)
        l = MaxPooling2D(l)

        l = tf.layers.dropout(l, rate=self.params.dropout_rate, training=is_training)

        l = inception_module(l, 16, [(3, 3), (5, 5), (7, 7), (1, 1)])
        l = BatchNormalization(l)

        l = ComplexBlock(l, 32)
        l = ComplexBlock(l, 64)

        def LatentConditional(input):
            with tf.variable_scope("Parallel"):
                parallel_layers = list()
                m, latent_size = latent_priors.shape
                for z in range(latent_size):
                    tf.variable_scope("Track%d" % z)
                    l = ComplexBlock(input, 128)

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
                        py_xzi = tf.sigmoid(logits, name="P(y|x,z=%d)" % z)
                        parallel_layers.append(py_xzi)

                # Conditional Probability of class, given X and latent variable(s)
                with tf.variable_scope("P(y|x,z)"):
                    py_xz = np.concatenate(parallel_layers, axis=1)
                return py_xz

        with tf.variable_scope("MarginalizeLatent"):
            pl_xc = LatentConditional(l)

            # Joint probability over class and latent
            with tf.variable_scope("P(y|x,z)P(z|x)"):
                pz_x = latent_priors
                pyz_x = np.multiply(pl_xc, pz_x, name="JointProbability")

            # Marginalize Latent
            py_x = tf.reduce_sum(pyz_x, axis=1)

        return py_x, f1_cost(py_x, labels)
