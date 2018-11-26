#!/usr/bin/env python
"""
File: data_set
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import csv

from HumanProteinAtlas import sample
from partitions import Split


class Dataset:

    def __init__(self, path, split=Split.train, cache=False):
        self.path = path
        self._cache = cache

        if split == Split.train:
            self.data_path = os.path.join(path, "train")
            self.csv_path = os.path.join(path, "train.csv")

        elif split == Split.test:
            self.data_path = os.path.join(path, "test")
            self.csv_path = os.path.join(path, "sample_submission.csv")

        self._shape = None
        self._sample_ids = None
        self._sample_labels = dict()
        self._samples = dict()

    @property
    def sample_ids(self):
        if self._sample_ids is None:
            self._sample_ids = list(self.sample_labels.keys())
        return self._sample_ids

    @property
    def sample_labels(self):
        if self._sample_labels:  # not an empty dictionary
            return self._sample_labels

        self._sample_labels = self.read_labels(self.csv_path)
        return self._sample_labels

    @property
    def shape(self):
        if self._shape is not None:
            return self._shape

        m = self.num_samples

        sample_shape = (4,) + self.sample(self.sample_ids[0]).shape
        self._shape = (m,) + sample_shape
        return self._shape

    def __len__(self):
        return len(self.sample_labels)

    @property
    def num_samples(self):
        return len(self.sample_ids)

    def sample(self, sample_id):
        if sample_id in self._samples:
            return self._samples[sample_id]

        if sample_id not in self.sample_labels:
            raise ValueError("Non-esitatnt sample: %s" % sample_id)

        labels = self.sample_labels[sample_id]
        s = sample.Sample(sample_id, labels, self.data_path, cache=self._cache)
        if self._cache:
            self._samples[sample_id] = s
        return s

    def __getitem__(self, sample_id):
        return self.sample(sample_id)

    def __iter__(self):
        return self

    def __next__(self):
        for sample_id in self.sample_ids:
            return self.sample(sample_id)

    def samples(self):
        # caution: this function reads every single sample's
        # meta-data into memory
        for sample_id in self.sample_ids:
            if sample_id not in self._samples:
                self._samples[sample_id] = self.sample(sample_id)
        return self._samples

    def drop_cache(self):
        for sample in self._samples.values():
            sample.drop_cache()

        self._samples = dict()
        self._sample_labels = dict()

    @staticmethod
    def read_labels(csv_file):
        d = dict()
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip the header
            for sample_id, label_list in reader:
                labels = list(map(int, label_list.split()))
                d[sample_id] = labels
        return d