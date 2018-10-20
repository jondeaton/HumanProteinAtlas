#!/usr/bin/env python
"""
File: data_set
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
from enum import Enum
import csv
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
        self._samples = dict()

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

        self._sample_labels = self.read_labels(self.csv_path)
        return self._sample_labels

    def sample(self, sample_id):
        if sample_id in

        s = sample.Sample(sample_id)

        return s

    

    @staticmethod
    def read_labels(csv_file):
        d = dict()
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for sample_id, label_list in reader:
                labels = list(map(int, label_list.split()))
                d[sample_id] = labels
        return d