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

def main():
    def main():
        args = parse_args()

        if args.google_cloud:
            logger.info("Running on Google Cloud.")

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
        logger.debug("BraTS data set directory: %s" % config.brats_directory)
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
    """
    Parse the command line options for this file
    :return: An argparse object containing parsed arguments
    """

    parser = argparse.ArgumentParser(description="Train human protein atlas model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options = parser.add_argument_group("Info")
    info_options.add_argument("--job-dir", default=None, help="Job directory")
    info_options.add_argument("--job-name", default="segmentation", help="Job name")
    info_options.add_argument("-params", "--params", type=str, help="Hyperparameters json file")
    info_options.add_argument("--config", required=True, type=str, help="Configuration file")

    input_options = parser.add_argument_group("Input")
    input_options.add_argument('--path', help="Training ")

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
