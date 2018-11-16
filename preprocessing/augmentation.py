#!/usr/bin/env python
"""
File: augmentation
Date: 10/21/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf


def augment_dataset(dataset):
    assert isinstance(dataset, tf.data.Dataset)

    flipped_lr = dataset.map(_flip_left_right)
    dataset = dataset.concatenate(flipped_lr)

    flipped_ud = dataset.map(_flip_up_down)
    dataset = dataset.concatenate(flipped_ud)

    # todo: more augmentation?

    return dataset


def _adjust(image, label):
    tf.image.random_brightness()


def _flip_left_right(image, label):
    with tf.variable_scope("random_flip_left_right"):
        return tf.image.flip_left_right(image), label


def _flip_up_down(image, label):
    with tf.variable_scope("random_flip_up_down"):
        return tf.image.flip_up_down(image), label
