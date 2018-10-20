#!/usr/bin/env python
"""
File: create
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os, sys
import logging
import argparse
import HumanProteinAtlas
import random

import partitions
from partitions import Split


def main():
    args = parse_args()
    input = os.path.expanduser(args.input)

    logger.debug("input: %s" % input)
    dataset = HumanProteinAtlas.Dataset(input)

    logger.info("Splitting dataset info train, test, validation...")
    test_ids = set(random.sample(dataset.sample_ids, args.num_test))

    remaining = [id for id in dataset.sample_ids if id not in test_ids]

    validation_ids = set(random.sample(remaining, args.num_validation))

    train_ids = [id for id in dataset.sample_ids
                 if id not in test_ids and id not in validation_ids]

    logger.info("Saving spilt ids to: %s" % partitions.default_partition_store)
    if not os.path.exists(partitions.default_partition_store):
        try:
            os.mkdir(partitions.default_partition_store)
        except FileExistsError:
            pass

    logger.info("Saving %d training ids..." % len(train_ids))
    save_split(train_ids, partitions.default_locations[Split.train])

    logger.info("Saving %d test ids..." % len(test_ids))
    save_split(test_ids, partitions.default_locations[Split.test])

    logger.info("Saving %d validation ids..." % len(validation_ids))
    save_split(validation_ids, partitions.default_locations[Split.validation])


def save_split(ids, output):
    with open(output, 'w') as f:
        f.write("\n".join(ids))


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset partitioning")

    split_options = parser.add_argument_group("Splits")
    split_options.add_argument("--num-test",
                               type=int,
                               required=True,
                               help="Number of test examples")

    split_options.add_argument("--num-validation",
                               type=int,
                               required=True,
                               help="Number of validation examples")

    input_options = parser.add_argument_group("Input")
    input_options.add_argument("-in", "--input",
                               type=str,
                               required=True,
                               help="Human Protein Atlas Dataset location")

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
