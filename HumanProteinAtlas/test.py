#!/usr/bin/env python
"""
File: test
Date: 10/20/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import unittest
import HumanProteinAtlas


class DataLoaderTest(unittest.TestCase):

    def test_ids(self):
        dataset = HumanProteinAtlas.Dataset()


if __name__ == "__main__":
    unittest.main()
