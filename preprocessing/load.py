#!/usr/bin/env python
"""
File: loading
Date: 10/21/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf

from HumanProteinAtlas import Dataset, Sample, Organelle
from partitions import Split, partitions


def load_dataset(dataset, split):
    """ Loads a dataset into a TensorFlow Dataset

    :param dataset: HumanProteinAtlas dataset
    :param split: which partition to use
    :return: TensorFlow Dataset made from the HumanProteinAtlas dataset
    """
    assert isinstance(dataset, Dataset)
    assert isinstance(split, Split)

    def sample_generator():
        for sample_id in partitions[split]:
            sample = dataset.sample(sample_id)
            assert isinstance(sample, Sample)
            yield sample.multi_channel, sample.multi_hot_label

    sample_shape_shape = dataset.shape[1:]
    label_shape = (len(Organelle),)

    return tf.data.Dataset.from_generator(sample_generator,
                                          output_types=(tf.float32, tf.float32),
                                          output_shapes=(tf.TensorShape(sample_shape_shape),
                                                         tf.TensorShape(label_shape)))
