#!/usr/bin/env python
"""
File: train
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os, sys
import argparse, logging

import tensorflow as tf

from deep_model.config import Configuration
from deep_model.params import Params
from deep_model.model_trainer import ModelTrainer


from deep_model.InceptionModel import InceptionModel
from deep_model.BaselineModel import BaselineModel
from deep_model.InceptionV1 import InceptionV1
from deep_model.InceptionFrozen import InceptionFrozen


from HumanProteinAtlas import Dataset, Organelle
from partitions import Split
from preprocessing import load_dataset, augment_dataset, preprocess_dataset

import numpy as np

# classes which we have only few examples of
rare_classes = [9, 10, 11, 16, 26, 27]


def create_datasets(human_protein_atlas):
    assert isinstance(human_protein_atlas, Dataset)

    datasets = [load_dataset(human_protein_atlas, split)
                for split in (Split.train, Split.test, Split.validation)]

    # any pre-processing to be done on all data sets
    for i, dataset in enumerate(datasets):
        datasets[i] = preprocess_dataset(datasets[i])

    train_dataset, test_dataset, validation_dataset = datasets

    rare_dataset = load_dataset(human_protein_atlas, Split.test, classes=rare_classes)
    rare_dataset = preprocess_dataset(rare_dataset)

    train_dataset = train_dataset.concatenate(rare_dataset)

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


def create_filtered_dataset(human_protein_atlas):
    assert isinstance(human_protein_atlas, Dataset)
    dataset = load_dataset(human_protein_atlas, Split.train)
    dataset = preprocess_dataset(dataset)


    rare_classes = [9, 10, 11, 16, 26, 27]
    rare_class_mask = np.zeros(len(Organelle))
    rare_class_mask[rare_classes] = 1

    def rare_class(image, label):
        is_rare = tf.cast(label * rare_class_mask, tf.bool)
        return tf.reduce_any(is_rare)

    dataset = dataset.filter(rare_class)

    dataset = dataset.shuffle(params.shuffle_buffer_size)
    dataset = dataset.batch(params.mini_batch_size)
    dataset = dataset.prefetch(buffer_size=params.prefetch_buffer_size)

    return dataset


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

    # Set random seed for reproducible results
    tf.set_random_seed(params.seed)

    logger.info("Creating data pre-processing pipeline...")
    logger.debug("Human Protein Atlas dataset: %s" % config.dataset_directory)

    human_protein_atlas = Dataset(config.dataset_directory)
    train_dataset, test_dataset, _ = create_datasets(human_protein_atlas)

    # filtered_dataset = create_filtered_dataset(human_protein_atlas)

    logger.info("Initiating training...")
    logger.debug("TensorBoard Directory: %s" % config.tensorboard_dir)
    logger.debug("Model save file: %s" % config.model_file)
    logger.debug("Num epochs: %s" % params.epochs)
    logger.debug("Mini-batch size: %s" % params.mini_batch_size)

    # model = InceptionFrozen(params)
    model = InceptionV1(params)
    trainer = ModelTrainer(model, config, params, logger, restore_model_path=args.restore)

    trainer.train(train_dataset, test_dataset,
                        trainable_scope=args.scope,
                        supplemental_dataset=None)

    logger.info("Exiting.")


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
