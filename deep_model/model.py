#!/usr/bin/env python
"""
File: model
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf


def model(input):

    is_training = tf.placeholder(tf.bool)
