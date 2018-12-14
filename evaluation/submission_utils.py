#!/usr/bin/env python
"""
File: reduce_features
Date: 12/7/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import os
import csv
import numpy as np

'''
Builds kaggle submission csv file for predictions dictionary
of the form: id:multi_hot_labels.

ex. "id1":[0, 5, 11]
'''
def build_kaggle_submission(predictions_dict, save_dir):
    assert isinstance(predictions_dict, dict)
    assert isinstance(save_dir, str)

    filepath = os.path.join(save_dir, "submission.csv")

    with open(filepath, "w+", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Ids", "Predicted"])

        for id, labels in predictions_dict.items():
            writer.writerow([str(id), " ".join(map(str, labels))])

    return filepath
