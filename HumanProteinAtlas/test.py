#!/usr/bin/env python
"""
File: test
Date: 10/20/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import unittest
import argparse

import HumanProteinAtlas
import numpy as np

# change this to be the path on your mahcine
hpa_dataset_path = os.path.expanduser("~/Datasets/HumanProteinAtlas")


class DataLoaderTest(unittest.TestCase):

    def test_path(self):
        dataset = HumanProteinAtlas.Dataset(hpa_dataset_path)
        self.assertEqual(dataset.path, hpa_dataset_path)

    def test_ids(self):
        dataset = HumanProteinAtlas.Dataset(hpa_dataset_path)
        self.assertIsInstance(dataset.sample_ids, list)

        for sample_id in dataset.sample_ids:
            self.assertIsInstance(sample_id, str)

    def test_labels(self):
        dataset = HumanProteinAtlas.Dataset(hpa_dataset_path)
        self.assertIsInstance(dataset.sample_labels, dict)

        for sample_id in dataset.sample_ids:
            self.assertIsInstance(dataset)

    def test_samples(self):
        dataset = HumanProteinAtlas.Dataset(hpa_dataset_path)

        for sample_id in dataset.sample_ids:
            sample = dataset.sample(sample_id)
            self.assertIsInstance(sample_id, HumanProteinAtlas.Sample, msg="Sample has correct name")
            self.assertEqual(sample_id, sample.id, msg="Requested sample has correct ID")

    def test_images(self):
        dataset = HumanProteinAtlas.Dataset(hpa_dataset_path)

        for sample_id in dataset.sample_ids:
            sample = dataset.sample(sample_id)

            for color in HumanProteinAtlas.Color:
                img = sample.image(color)
                self.assertIsInstance(img, np.ndarray)


if __name__ == "__main__":
    unittest.main()
