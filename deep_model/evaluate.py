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


def evaluate_model(dataset, ids, output_dir, recompute=False, run_model=None, name="test"):
    """
    Evaluate the performance of a prediction model on a collection of samples

    :param dataset: HumanProteinAtlas Dataset
    :param ids: List of sample IDs from the dataset on which to evaluate the model
    :param output_dir: output directory to write outputs to
    :param recompute: Whether to recompute the predictions
    :param run_model: Function which will run predictions on a model
    :param name: The name of this evaluation
    :return: None
    """
    assert isinstance(dataset, HumanProteinAtlas.Dataset)

    if recompute and run_model is None:
        raise ValueError("Must pass model to run_model if recomputing predictions")

    predictions_file = os.path.join(output_dir, "%s_preds.npy" % name)
    if not recompute and not os.path.exists(predictions_file):
        raise ValueError("Predictions file not found: %s" % predictions_file)

    if recompute or not os.path.exists(predictions_file):
        logger.info("Computing predictions...")
        y_score = compute_predictions(dataset, run_model, ids)
        logger.info("Saving predictions to: %s" % predictions_file)
        np.save(predictions_file, y_score)
    else:
        # if not recomputing... just trying to reload them from a previous save file
        logger.info("Loading pre-computed predictions from: %s" % predictions_file)
        y_score = np.load(predictions_file)

    logger.info("Creating evaluation metrics report...")
    y_true = labels_matrix(dataset, ids)
    y_pred = np.where(y_score >= 0.5, 1, 0)

    metrics.create_report(y_true, y_score, y_pred, output_dir, print=True)


def compute_predictions(dataset, run_model, ids, batch_size=64):
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


def restore_model(save_path, model_file):
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

        return run_model


def labels_matrix(dataset, ids):
    m = len(ids)
    labels = np.empty((m, len(Organelle)))
    for i, id in enumerate(ids):
        sample = dataset.sample(id)
        labels[i, :] = sample.multi_hot_label
    return labels


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

    dataset = HumanProteinAtlas.Dataset(config.dataset_directory, scale=args.scale)
    run_model = None
    if args.recompute:
        logger.info("Restoring model...")
        run_model = restore_model(save_path, model_file)
        logger.info("Model successfully restored.")

    test_output = os.path.join(output_dir, "test_evaluation")
    if not os.path.isdir(test_output):
        os.makedirs(test_output, exist_ok=True)

    evaluate_model(dataset, partitions.test, test_output,
                   recompute=args.recompute, run_model=run_model, name="test")

    valid_output = os.path.join(output_dir, "validation_evaluation")
    if not os.path.isdir(valid_output):
        os.makedirs(valid_output, exist_ok=True)
    evaluate_model(dataset, partitions.validation, valid_output,
                   recompute=args.recompute, run_model=run_model, name="validation")


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
