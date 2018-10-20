#!/usr/bin/env python
"""
File: partition
Date: 10/20/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
from enum import Enum


class Split(Enum):
    train = 0

    # yes, these two are set to the same thing because these
    # two terms refer to the same thing (i.e. use either one)
    test = 1
    dev = 1

    validation = 2

# this class represents the collection of ids that belong to the
# train, test, validaiton sets
class Partitions:
    def __init__(self, locations):
        self._locations = locations
        self._partitions = dict()

    def __getitem__(self, split):
        assert isinstance(split, Split)
        if split not in self._partitions:
            split_file = self._locations[split]
            self._partitions[split] = self._get_ids(split_file)
        return self._partitions[split]

    @property
    def train(self):
        return self[Split.train]

    @property
    def test(self):
        return self[Split.test]

    @property
    def validation(self):
        return self[Split.validation]

    @staticmethod
    def _get_ids(split_file):
        if os.path.isfile(split_file):
            with open(split_file, 'r') as f:
                return f.readlines()
