#!/usr/bin/env python
"""
File: loading
Date: 10/21/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf

from HumanProteinAtlas import Dataset, Sample
from partitions import Split, partitions


def load_dataset(dataset, split):
    """ Loads a dataset into a TensorFlow

    :param dataset:
    :param split:
    :return:
    """
    assert isinstance(dataset, Dataset)
    assert isinstance(split, Split)

    def sample_generator():
        for sample_id in partitions[split]:
            sample = dataset.sample(sample_id)
            assert isinstance(sample, Sample)
            yield sample.multi_channel, sample.labels

    return tf.data.Dataset.from_generator(sample_generator)
