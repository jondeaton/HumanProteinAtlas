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

import tensorflow as tf

from deep_model.config import Configuration
from deep_model.params import Params


def train(train_dataset, test_dataset):
    """
    Train the model
    :param train_dataset: Training dataset
    :param test_dataset: Testing/dev dataset
    :return: None
    """

    # Set up dataset iterators
    dataset_handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(dataset_handle,
                                                   train_dataset.output_types,
                                                   train_dataset.output_shapes)

    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    # MRI input and ground truth segmentations
    input, seg = iterator.get_next()
    input = tf.identity(input, "input")

    # Create the model's computation graph and cost function
    logger.info("Instantiating model...")
    output, is_training = UNet.model(input, seg, params.multi_class, params.patch)
    output = tf.identity(output, "output")

    '''if params.patch:
        output = _to_patch_prediction(output)'''

    if params.multi_class:
        pred = _to_prediction(output, params.multi_class)
        dice = multi_class_dice(seg, pred)
    else:
        dice = dice_coeff(seg, output)

    # Cost function
    if params.loss == loss.dice:
        cost = - dice
    else:
        x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=seg, logits=output)
        cost = tf.reduce_mean(x_entropy)

    # Define the optimization strategy
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer, learning_rate = _get_optimizer(cost, global_step)

    logger.info("Training...")
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # Configure TensorBoard training data
        train_dice = tf.summary.scalar('train_dice', dice)
        train_dice_histogram = tf.summary.histogram("train_dice_histogram", dice)
        train_dice_average = tf.summary.scalar('train_dice_average', tf.reduce_mean(dice))
        train_cost = tf.summary.scalar('train_cost', cost)
        merged_summary_train = tf.summary.merge([train_dice, train_dice_histogram, train_dice_average, train_cost])

        # Configure TensorBoard test data
        test_dice = tf.summary.scalar('test_dice', dice)
        test_dice_histogram = tf.summary.histogram('test_dice_histogram', dice)
        test_dice_average = tf.summary.scalar('test_dice_average', tf.reduce_mean(dice))
        test_cost = tf.summary.scalar('test_cost', cost)
        merged_summary_test = tf.summary.merge([test_dice, test_dice_histogram, test_dice_average, test_cost])

        writer = tf.summary.FileWriter(logdir=tensorboard_dir)
        writer.add_graph(sess.graph)  # Add the pretty graph viz

        # Initialize graph, data iterators, and model saver
        sess.run(init)
        train_handle = sess.run(train_iterator.string_handle())
        saver = tf.train.Saver(save_relative_paths=True)
        saver.save(sess, config.model_file, global_step=global_step)

        # frequency (number of batches) after which we display test error
        tb_freq = np.round(config.tensorboard_freq / params.mini_batch_size)

        # Training epochs
        for epoch in range(params.epochs):
            sess.run(train_iterator.initializer)

            # Iterate through all batches in the epoch
            batch = 0

            while True:
                try:
                    train_summary, _, c, d = sess.run([merged_summary_train, optimizer, cost, dice],
                                                      feed_dict={is_training: True,
                                                                 dataset_handle: train_handle})

                    logger.info("Epoch: %d, Batch %d: cost: %f, dice: %f" % (epoch, batch, c, d))
                    writer.add_summary(train_summary, global_step=sess.run(global_step))

                    batch += 1
                    if batch % tb_freq == 0:
                        logger.info("logging test output to TensorBoard")
                        # Generate stats for test dataset
                        sess.run(test_iterator.initializer)
                        test_handle = sess.run(test_iterator.string_handle())

                        test_summary, test_avg = sess.run([merged_summary_test, test_dice_average],
                                                          feed_dict={is_training: False,
                                                                     dataset_handle: test_handle})
                        writer.add_summary(test_summary, global_step=sess.run(global_step))

                except tf.errors.OutOfRangeError:
                    logger.info("End of epoch %d" % epoch)
                    logger.info("Saving model...")
                    saver.save(sess, config.model_file, global_step=global_step)
                    logger.info("Model save complete.")
                    break

        logger.info("Training complete.")


def main():
    args = parse_args()

    global config
    config = Configuration(args.config)

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
    logger.debug("Human Protein Atlas dataset: %s" % config.brats_directory)
    logger.debug("TFRecords: %s" % config.tfrecords_dir)

    # get patch indices
    '''if params.patch:
        patch_indices = get_patch_indices(params.patches_per_image, mri_shape, params.patch_shape, seg)'''

    train_dataset, test_dataset, validation_dataset = create_data_pipeline(params.multi_class)

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
    info_options.add_argument("--config", required=True, type=str, help="Configuration file")

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
