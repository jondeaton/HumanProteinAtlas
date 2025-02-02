#!/usr/bin/env python
"""
File: loading
Date: 10/21/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf

from HumanProteinAtlas import Dataset, Sample, Organelle
from partitions import Split, partitions


def load_dataset(dataset, split, classes=None):
    """ Loads a dataset into a TensorFlow Dataset object

    :param dataset: HumanProteinAtlas dataset
    :param split: which partition to use
    :return: TensorFlow Dataset made from the HumanProteinAtlas dataset
    """
    assert isinstance(dataset, Dataset)
    assert isinstance(split, Split)

    if classes is not None:
        classes = set(classes)

    def sample_generator():
        for sample_id in partitions[split]:
            sample = dataset.sample(sample_id)
            if classes is None or any(l in classes for l in sample.labels):
                yield sample.multi_channel, sample.multi_hot_label

    sample_shape_shape = dataset.shape[1:]
    label_shape = (len(Organelle),)

    output_types = (tf.float32, tf.float32)
    output_shapes = (tf.TensorShape(sample_shape_shape), tf.TensorShape(label_shape))
    return tf.data.Dataset.from_generator(sample_generator, output_types=output_types, output_shapes=output_shapes)


def load_gmm_dataset(human_protein_atlas, split, get_gmm_probabilities, gmm_num_latent_classes, classes=None):
    assert isinstance(human_protein_atlas, Dataset)
    assert isinstance(split, Split)

    if classes is not None:
        classes = set(classes)

    sample_shape_shape = human_protein_atlas.shape[1:]
    label_shape = (len(Organelle),)
    probas_shape = (gmm_num_latent_classes,)

    def sample_generator():
        for sample_id in partitions[split]:
            sample = human_protein_atlas.sample(sample_id)
            if classes is None or any(l in classes for l in sample.labels):
                gmm_probas = get_gmm_probabilities(sample)
                yield sample.multi_channel, gmm_probas, sample.multi_hot_label

    output_types = (tf.float32, tf.float32, tf.float32)
    output_shapes = (tf.TensorShape(sample_shape_shape), tf.TensorShape(probas_shape), tf.TensorShape(label_shape))
    dataset = tf.data.Dataset.from_generator(sample_generator, output_types=output_types, output_shapes=output_shapes)
    return dataset
