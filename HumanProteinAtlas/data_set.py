#!/usr/bin/env python
"""
File: data_set
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
from enum import Enum
from HumanProteinAtlas import sample

class Split(Enum):
    test = 0
    train = 0

class Dataset:

    def __init__(self, path, split=Split.train, cache=False):
        self.path = path
        self.cache = False

        if split == Split.train:
            self.data_path = os.path.join(path, "train")
            self.csv_path = os.path.join(path, "train.csv")
        elif split == Split.test:
            self.data_path = os.path.join(path, "test")
            self.csv_path = os.path.join(path, "sample_submission.csv")

        self._sample_labels = None

    @property
    def labels(self):
        return

    @property
    def sample_ids(self):
        return self.sample_labels.keys()

    @property
    def sample_labels(self):
        if self._sample_labels is not None:
            return self._sample_labels

        self._sample_labels =
        return self._sample_labels


    @property
    def sample(self, sample_id):

        sample.Sample(sample_id)