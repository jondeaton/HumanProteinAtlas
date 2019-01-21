#!/usr/bin/env python
"""
File: train
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os, sys
import argparse, logging

import numpy as np
import tensorflow as tf

from deep_model.config import Configuration
from deep_model.params import Params
from deep_model.model_trainer import ModelTrainer

from deep_model.InceptionV1 import InceptionV1

from HumanProteinAtlas import Dataset
from partitions import Split
from preprocessing.load import load_dataset
from preprocessing.augmentation import augment_dataset
from preprocessing.preprocess import preprocess_dataset


def create_datasets(human_protein_atlas, bias_rare=False, rare_classes=None):
    assert isinstance(human_protein_atlas, Dataset)

    if bias_rare and rare_classes is None:
        raise ValueError("Provide list of rare classes to use rare class bias")

    splits = (Split.train, Split.test, Split.validation)
    datasets = [load_dataset(human_protein_atlas, split) for split in splits]

    if bias_rare:
        rare_dataset = load_dataset(human_protein_atlas, Split.train, classes=rare_classes)
        datasets[0] = datasets[0].concatenate(rare_dataset)  # add it into the training dataset

    # any pre-processing to be done across all data sets
    for i, dataset in enumerate(datasets):
        datasets[i] = preprocess_dataset(datasets[i])

    train_dataset, test_dataset, validation_dataset = datasets

    # Optional Training Dataset augmentation
    if params.augment:
        train_dataset = augment_dataset(train_dataset)

    # Shuffle, batch, prefetch the training data set
    train_dataset = train_dataset.shuffle(params.shuffle_buffer_size)
    train_dataset = train_dataset.batch(params.mini_batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=params.prefetch_buffer_size)

    # Shuffle/batch test data set
    test_dataset = test_dataset.shuffle(params.shuffle_buffer_size)
    test_dataset = test_dataset.batch(params.test_batch_size)

    return train_dataset, test_dataset, validation_dataset


def run_train(args, config, params):

    np.random.seed(params.seed)
    tf.set_random_seed(params.seed)   # Set random seed for reproducible results

    logger.info("Creating data pre-processing pipeline...")
    logger.debug("Human Protein Atlas dataset: %s" % config.dataset_directory)

    human_protein_atlas = Dataset(config.dataset_directory)
    train_dataset, test_dataset, _ = create_datasets(human_protein_atlas,
                                                     bias_rare=params.bias_rare,
                                                     rare_classes=params.rare_classes)

    logger.info("Initiating training...")
    logger.debug("TensorBoard Directory: %s" % config.tensorboard_dir)
    logger.debug("Model save file: %s" % config.model_file)
    logger.debug("Num epochs: %s" % params.epochs)
    logger.debug("Mini-batch size: %s" % params.mini_batch_size)

    model = InceptionV1(params)
    trainer = ModelTrainer(model, config, params, logger, restore_model_path=args.restore)
    trainer.train(train_dataset, test_dataset, trainable_scopes=params.scopes)

    logger.debug("Exiting.")


def main():
    args = parse_args()

    global config
    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration()  # use default

    global params
    if args.params is not None:
        params = Params(args.params)
    else:
        params = Params()

    params.override(args)
    run_train(args, config, params)


def parse_args():
    parser = argparse.ArgumentParser(description="Train human protein atlas model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_groups = parser.add_argument_group("Input")
    input_groups.add_argument('--path', help="Dataset input file")

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--model-file", help="File to save trained model in")

    info_group = parser.add_argument_group("Info")
    info_group.add_argument("-params", "--params", type=str, help="Hyperparameters json file")
    info_group.add_argument("--config", required=False, type=str, help="Configuration file")

    restore_group = parser.add_argument_group("Restore")
    restore_group.add_argument("--restore", type=str, required=False, help="Model to restore and continue training")

    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--epochs", type=int, required=False, help="Number of epochs to train")
    training_group.add_argument("--scope", type=str, required=False, help="Trainable variable scope")

    tensorboard_group = parser.add_argument_group("TensorBoard")
    tensorboard_group.add_argument("--tensorboard", help="TensorBoard directory")
    tensorboard_group.add_argument("--log-frequency", help="Logging frequency")

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument('--log',
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
