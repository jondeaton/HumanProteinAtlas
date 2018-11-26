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


def multi_hot_to_list(multi_hot, threshold=0.5):
    l = list()
    for i in range(len(multi_hot)):
        if multi_hot[i] > threshold:
            l.append(i)
    return l

def evaluate_on(run_model, test_ids, output_dir):
    dataset = HumanProteinAtlas.Dataset(config.dataset_directory)

    m = len(test_ids)
    true_labels = np.empty((m, len(Organelle)))
    probabilities = np.empty((m, len(Organelle)))

    batch = np.empty((16, len(Organelle)))

    for i, id in enumerate(test_ids):
        if i == m: break

        sample = dataset.sample(id)

        img = np.expand_dims(sample.multi_channel, axis=0)
        probabilities[i, :] = run_model(img)

        if i % 100 == 0:
            logger.info("Done with %d samples." % i)

        true_labels[i, :] = sample.multi_hot_label

    output_file = os.path.join(output_dir, "metrics.txt")
    metrics.evaluation_metrics(true_labels, probabilities, output_file=output_file)

def evaluate(run_model, output_dir):

    train_ids = partitions[Split.train]
    test_ids = partitions[Split.test]
    validation_ids = partitions[Split.validation]

    logger.info("Evaluating test data...")
    evaluate_on(run_model, test_ids, output_dir)

    logger.info("Evaluating validation data...")
    evaluate_on(run_model, validation_ids, output_dir)

    logger.info("Evaluating training data...")
    evaluate_on(run_model, train_ids, output_dir)


def restore_and_evaluate(save_path, model_file, output_dir):
    tf.reset_default_graph()

    with tf.Session() as sess:

        logger.info("Restoring model: %s" % model_file)
        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, tf.train.latest_checkpoint(save_path))
        logger.info("Model restored.")

        graph = tf.get_default_graph()

        input = graph.get_tensor_by_name("input:0")
        output = graph.get_tensor_by_name("output:0")
        is_training = graph.get_tensor_by_name("Placeholder_1:0")

        def run_model(sample):
            feed_dict = {input: sample, is_training: False}
            return sess.run(output, feed_dict=feed_dict)

        logger.info("Evaluating mode...")
        evaluate(run_model, output_dir)


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

    restore_and_evaluate(save_path, model_file, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the human protein atlas model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_options = parser.add_argument_group("Input")
    input_options.add_argument("--save-path", required=True, help="Tensorflow save path")
    input_options.add_argument("--model", required=True, help="File to save trained model in")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("-o", "--output", required=True, help="Output directory to store plots")

    info_options = parser.add_argument_group("Info")
    info_options.add_argument("--config", required=False, type=str, help="Configuration file")
    info_options.add_argument("-params", "--params", type=str, help="Hyperparameters json file")

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
