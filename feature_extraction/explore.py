#!/usr/bin/env python
"""
File: explore
Date: 12/7/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

#!/usr/bin/env python
"""
File: extract
Date: 12/7/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os, sys
import logging, argparse

import numpy as np

import HumanProteinAtlas
from deep_model.config import Configuration
from HumanProteinAtlas import Dataset, Color

from partitions import partitions
from feature_extraction import get_features

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from feature_extraction import Feature
import multiprocessing as mp

def explore_features(X):
    pca = PCA(n_components=2, whiten=True)
    principalComponents = pca.fit_transform(X)

    plt.figure()
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
    plt.show()


def _get_ft(t):
    return get_ft(*t)


def get_ft(dataset, id):
    logger.info("Extracting features for: %s" % id)
    img = dataset[id].combined((Color.blue, Color.yellow, Color.red))
    return get_features(img, method=Feature.dct)


def main():
    args = parse_args()

    global config
    if args.config is not None:
        config = Configuration(args.config)
    else:
        config = Configuration()

    if args.output is not None:
        output_dir = os.path.expanduser(args.output)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    if not os.path.exists("X.npy"):
        dataset = HumanProteinAtlas.Dataset(config.dataset_directory, scale=True)
        experimental_ids = np.random.choice(partitions.train, args.num_examples, replace=False)

        arguments = [(dataset, id) for id in experimental_ids]
        pool = mp.Pool(8)
        features = pool.map(_get_ft, arguments)

        X = np.vstack(features)

        np.save("X", X)
    else:
        X = np.load("X.npy")

    explore_features(X)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Features",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("-o", "--output", required=False, help="Output directory to store plots")

    options_group = parser.add_argument_group("Options")
    options_group.add_argument("--config", required=False, type=str, help="Configuration file")
    options_group.add_argument("-p", "--pool-size", type=int, default=8, help="Worker pool size")
    options_group.add_argument("-m", "--num-examples", type=int, default=2000, help="Number of examples")

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
