#!/usr/bin/env python
"""
File: evluate
Date: 11/15/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import sys
import logging
import argparse

import numpy as np
import tensorflow as tf

import HumanProteinAtlas
from deep_model.config import Configuration
from partitions import partitions, Split
from evaluation import metrics
from HumanProteinAtlas import Organelle

# Solves OpenMP multiple installation problem
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def multi_hot_to_list(multi_hot, threshold=0.5):
    l = list()
    for i in range(len(multi_hot)):
        if multi_hot[i] > threshold:
            l.append(i)
    return l


def labels_matrix(dataset, ids):
    m = len(ids)
    labels = np.empty((m, len(Organelle)))
    for i, id in enumerate(ids):
        sample = dataset.sample(id)
        labels[i, :] = sample.multi_hot_label
    return labels


def evaluate_on(dataset, run_model, ids, output_dir, name=""):
    predictions_file = os.path.join(output_dir, "%s_preds.npy" % name)
    output_file = os.path.join(output_dir, "metrics.txt")

    if recompute or not os.path.exists(predictions_file):
        probabilities = get_predictions(dataset, run_model, ids)
        np.save(predictions_file, probabilities)
    else:
        probabilities = np.load(predictions_file)

    true_labels = labels_matrix(dataset, ids)
    metrics.evaluation_metrics(true_labels, probabilities, output_file=output_file)


def get_predictions(dataset, run_model, ids, batch_size=64):
    m = len(ids)
    probabilities = np.empty((m, len(Organelle)))

    batch = np.empty((batch_size,) + dataset.image_shape)

    for i, id in enumerate(ids):
        sample = dataset.sample(id)
        batch[i % batch_size, :] = sample.multi_channel

        if i % batch_size == batch_size - 1:
            start = i - batch_size + 1
            end = i + 1
            probabilities[start:end, :] = run_model(batch)

        if i % 100 == 0:
            logger.info("Done with %d samples." % i)


    return probabilities


def evaluate(dataset, run_model, output_dir, recompute=False):
    logger.info("Evaluating test data...")
    evaluate_on(dataset, run_model, partitions.test, output_dir, name="test")

    logger.info("Evaluating validation data...")
    evaluate_on(dataset, run_model, partitions.validation, output_dir, name="valid")


def restore_and_evaluate(dataset, save_path, model_file, output_dir):
    tf.reset_default_graph()

    with tf.Session() as sess:

        logger.info("Restoring model: %s" % model_file)
        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, tf.train.latest_checkpoint(save_path))

        graph = tf.get_default_graph()

        input = graph.get_tensor_by_name("input:0")
        output = graph.get_tensor_by_name("output:0")
        is_training = graph.get_tensor_by_name("Placeholder_1:0")

        logger.info("Model restored.")

        def run_model(samples):
            feed_dict = {input: samples, is_training: False}
            return sess.run(output, feed_dict=feed_dict)

        logger.info("Evaluating model...")
        evaluate(dataset, run_model, output_dir)


def main():
    args = parse_args()

    global config
    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration()

    save_path = os.path.expanduser(args.save_path)
    if not os.path.isdir(save_path):
        logger.error("No such save-path directory: %s" % save_path)
        return

    model_file = os.path.join(save_path, args.model)
    if not os.path.exists(model_file):
        logger.error("No such file: %s" % model_file)
        return

    output_dir = os.path.expanduser(args.output)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    global recompute
    recompute = args.recompute

    dataset = HumanProteinAtlas.Dataset(config.dataset_directory, scale=args.scale)
    restore_and_evaluate(dataset, save_path, model_file, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the human protein atlas model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--save-path", required=True, help="Tensorflow save path")
    input_group.add_argument("--model", required=True, help="File to save trained model in")

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("-o", "--output", required=True, help="Output directory to store plots")

    options_group = parser.add_argument_group("Options")
    options_group.add_argument("--config", required=False, type=str, help="Configuration file")
    options_group.add_argument("-params", "--params", type=str, help="Hyperparameters json file")
    options_group.add_argument("--scale", action='store_true', help="Scale the images down")
    options_group.add_argument("--recompute", action='store_true', help="Recompute predictions")

    logging_options = parser.add_argument_group("Logging")
    logging_options.add_argument('--log', dest="log_level", default="DEBUG", help="Logging level")

    args = parser.parse_args()

    # Setup the logger
    global logger
    logger = logging.getLogger('root')

    # Logging level configuration
    log_level = getattr(logging, args.log_level.upper())
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)

    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')

    # For the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(log_level)

    return args


if __name__ == "__main__":
    main()
