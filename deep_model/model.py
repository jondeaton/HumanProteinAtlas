#!/usr/bin/env python
"""
File: model
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf


def model(input, labels):

    is_training = tf.placeholder(tf.bool)


    multi_hot_label = tf.multi
