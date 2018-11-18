#!/usr/bin/env python
"""
File: train
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import sys
import argparse
import logging
import datetime

import numpy as np
import tensorflow as tf

import deep_model
from deep_model.config import Configuration
from deep_model.params import Params

from HumanProteinAtlas import Dataset
from partitions import Split
from preprocessing import load_dataset, augment_dataset, preprocess_dataset


def train(train_dataset, test_dataset):
    assert isinstance(train_dataset, tf.data.Dataset)
    assert isinstance(test_dataset, tf.data.Dataset)

    # dataset iterators (for selecting dataset to feed in)
    dataset_handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(dataset_handle,
                                                   train_dataset.output_types,
                                                   train_dataset.output_shapes)

    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    input, labels = iterator.get_next()
    input = tf.identity(input, "input")

    # Create the model's computation graph and cost function
    logger.info("Instantiating model...")
    output, is_training = deep_model.model(input, labels, params=params)
    output = tf.identity(output, "output")

    # Cost function
    with tf.variable_scope("cost"):
        if params.cost == "unweighted":
            xS = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                         logits=output)
        else:
            # weighted cross-entropy
            xS = tf.nn.weighted_cross_entropy_with_logits(targets=labels,
                                                          logits=output,
                                                          pos_weight=params.positive_weight)

        cost = tf.reduce_mean(xS)

    logits = tf.sigmoid(output)
    correct_prediction = tf.equal(output, tf.round(labels))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    positive_correct = tf.equal(correct_prediction, tf.cast(labels, tf.bool))
    total_positive_correct = tf.reduce_sum(tf.cast(positive_correct, tf.float32))
    total_positive = tf.reduce_sum(tf.cast(labels, tf.float32))
    positive_accuracy = total_positive_correct / total_positive

    # Define the optimization strategy
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer, learning_rate = _get_optimizer(cost, global_step)

    logger.info("Training...")
    init = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()

    with tf.Session() as sess:

        # Configure TensorBoard training data
        # some_other_metric = tf.summary.scalar('train_dice', dice)
        train_cost = tf.summary.scalar('train_cost', cost)
        merged_summary_train = tf.summary.merge([train_cost])

        # Configure TensorBoard test data
        test_cost = tf.summary.scalar('test_cost', cost)
        test_accuracy = tf.summary.scalar('test_accuracy', accuracy)
        test_pos_accuracy = tf.summary.scalar('test_positive_accuracy', positive_accuracy)
        merged_summary_test = tf.summary.merge([test_cost, test_accuracy, test_pos_accuracy])

        writer = tf.summary.FileWriter(logdir=tensorboard_dir)
        writer.add_graph(sess.graph)  # Add the pretty graph viz

        # Initialize graph, data iterators, and model saver
        sess.run(init)
        sess.run(init_l)
        train_handle = sess.run(train_iterator.string_handle())
        saver = tf.train.Saver(save_relative_paths=True)
        saver.save(sess, config.model_file, global_step=global_step)

        # frequency (number of batches) after which we display test error
        tb_freq = np.round(config.tensorboard_freq)

        # Training epochs
        for epoch in range(params.epochs):
            sess.run(train_iterator.initializer)

            # Iterate through all batches in the epoch
            batch = 0

            while True:
                try:
                    train_summary, _, c = sess.run([merged_summary_train, optimizer, cost],
                                                      feed_dict={is_training: True,
                                                                 dataset_handle: train_handle})

                    logger.info("Epoch: %d, Batch %d: cost: %f" % (epoch, batch, c))
                    writer.add_summary(train_summary, global_step=sess.run(global_step))

                    batch += 1
                    if batch % tb_freq == 0:
                        logger.info("Logging test output to TensorBoard")
                        # Generate stats for test dataset
                        sess.run(test_iterator.initializer)
                        test_handle = sess.run(test_iterator.string_handle())

                        test_summary = sess.run(merged_summary_test,
                                                    feed_dict={is_training: False,
                                                                dataset_handle: test_handle})

                        acc = sess.run(accuracy, feed_dict={is_training: False,
                                                            dataset_handle: test_handle})

                        pos_acc = sess.run(positive_accuracy,
                                           feed_dict={is_training: False,
                                                      dataset_handle: test_handle})

                        logger.info("Test accuracy: %s, Positive Accuracy: %s" % (acc, positive_accuracy))
                        writer.add_summary(test_summary, global_step=sess.run(global_step))

                    if batch % config.save_freq == 0:
                        logger.info("Saving model...")
                        saver.save(sess, config.model_file, global_step=global_step)
                        logger.info("Model save complete.")

                except tf.errors.OutOfRangeError:
                    logger.info("End of epoch %d" % epoch)
                    logger.info("Saving model...")
                    saver.save(sess, config.model_file, global_step=global_step)
                    logger.info("Model save complete.")
                    break

        logger.info("Training complete.")


def create_data_pipeline(human_protein_atlas):
    datasets = [load_dataset(human_protein_atlas, split)
                for split in (Split.train, Split.test, Split.validation)]

    for i, dataset in enumerate(datasets):
        datasets[i] = preprocess_dataset(datasets[i])

    train_dataset, test_dataset, validation_dataset = datasets

    # Dataset augmentation
    if params.augment:
        train_dataset = augment_dataset(train_dataset)

    # Shuffle, repeat, batch, prefetch the training dataset
    train_dataset = train_dataset.shuffle(params.shuffle_buffer_size)
    train_dataset = train_dataset.batch(params.mini_batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=params.prefetch_buffer_size)

    # Shuffle/batch test dataset
    test_dataset = test_dataset.shuffle(params.shuffle_buffer_size)
    test_dataset = test_dataset.batch(params.test_batch_size)

    return train_dataset, test_dataset, validation_dataset


def _get_optimizer(cost, global_step):
    if params.adam:
        # With Adam optimization: no learning rate decay
        learning_rate = tf.constant(params.learning_rate, dtype=tf.float32)
        sgd = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam")
    else:
        # Set up Stochastic Gradient Descent Optimizer with exponential learning rate decay
        learning_rate = tf.train.exponential_decay(params.learning_rate, global_step=global_step,
                                                   decay_steps=100000, decay_rate=params.learning_decay_rate,
                                                   staircase=False, name="learning_rate")
        sgd = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="SGD")
    optimizer = sgd.minimize(cost, name='optimizer', global_step=global_step)
    return optimizer, learning_rate


def _get_job_name():
    now = '{:%Y-%m-%d.%H:%M}'.format(datetime.datetime.now())
    return "%s_lr_%.4f" % (now, params.learning_rate)


def main():
    args = parse_args()

    global config
    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration() # use default

    global params
    if args.params is not None:
        params = Params(args.params)
    else:
        params = Params()

    # Set the TensorBoard directory
    global tensorboard_dir
    tensorboard_dir = os.path.join(config.tensorboard_dir, _get_job_name())

    # Set random seed for reproducible results
    tf.set_random_seed(params.seed)

    logger.info("Creating data pre-processing pipeline...")
    logger.debug("Human Protein Atlas dataset: %s" % config.dataset_directory)

    human_protein_atlas = Dataset(config.dataset_directory)

    train_dataset, test_dataset, _ = create_data_pipeline(human_protein_atlas)

    logger.info("Initiating training...")
    logger.debug("TensorBoard Directory: %s" % config.tensorboard_dir)
    logger.debug("Model save file: %s" % config.model_file)
    logger.debug("Learning rate: %s" % params.learning_rate)
    logger.debug("Num epochs: %s" % params.epochs)
    logger.debug("Mini-batch size: %s" % params.mini_batch_size)
    train(train_dataset, test_dataset)

    logger.info("Exiting.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train human protein atlas model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options = parser.add_argument_group("Info")
    info_options.add_argument("-params", "--params", type=str, help="Hyperparameters json file")
    info_options.add_argument("--config", required=False, type=str, help="Configuration file")

    input_options = parser.add_argument_group("Input")
    input_options.add_argument('--path', help="Dataset input file")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--model-file", help="File to save trained model in")

    tensorboard_options = parser.add_argument_group("TensorBoard")
    tensorboard_options.add_argument("--tensorboard", help="TensorBoard directory")
    tensorboard_options.add_argument("--log-frequency", help="Logging frequency")

    logging_options = parser.add_argument_group("Logging")
    logging_options.add_argument('--log',
                                 dest="log_level",
                                 choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                                 default="DEBUG", help="Logging level")

    args = parser.parse_args()

    # Setup the logger
    global logger
    logger = logging.getLogger('root')

    # Logging level configuration
    log_level = getattr(logging, args.log_level.upper())
    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')

    # For the console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(log_level)

    return args


if __name__ == "__main__":
    main()
