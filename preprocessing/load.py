#!/usr/bin/env python
"""
File: loading
Date: 10/21/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf

from HumanProteinAtlas import Dataset, Sample, Organelle
from partitions import Split, partitions

from HumanProteinAtlas.sample import Color, default_color_ordering


def load_dataset(dataset, split):
    assert isinstance(dataset, Dataset)
    assert isinstance(split, Split)

    def _load_image(sample_id):
        base = tf.string_join([dataset.data_path, "/", sample_id])

        color_filenames = [None] * len(Color)
        for color, idx in default_color_ordering.items():
            color_filenames[idx] = tf.string_join([base, "_{color}.png".format(color=color)])

        color_images = list()
        for filename in color_filenames:
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_png(image_string)
            color_images.append(image_decoded)

        full_image = tf.concat(color_images, axis=0)
        assert isinstance(full_image, tf.Tensor)
        full_image.set_shape(dataset.image_shape)
        return tf.cast(full_image, tf.float32)

    def _encode_label(labels):
        expanded_encoding = tf.one_hot(indices=labels, depth=len(Organelle))
        label_encoding = tf.reduce_sum(expanded_encoding, axis=0)
        return label_encoding

    ids = partitions[split]
    labels = [dataset.sample_labels[id] for id in ids]

    id_dataset = tf.data.Dataset.from_tensor_slices(ids)
    assert isinstance(id_dataset, tf.data.Dataset)
    image_dataset = id_dataset.map(_load_image)

    label_dataset = tf.data.Dataset.from_generator(lambda: labels, tf.int32, output_shapes=[None])
    assert isinstance(label_dataset, tf.data.Dataset)
    label_dataset = label_dataset.map(_encode_label)

    sample_shape = dataset.shape[1:]
    label_shape = (len(Organelle),)

    ds = tf.data.Dataset.zip((image_dataset, label_dataset))
    return ds


def _load_dataset(dataset, split, classes=None):
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

    sample_shape = dataset.shape[1:]
    label_shape = (len(Organelle),)

    output_types = (tf.float32, tf.float32)
    output_shapes = (tf.TensorShape(sample_shape), tf.TensorShape(label_shape))
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
