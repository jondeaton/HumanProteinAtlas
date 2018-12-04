#!/usr/bin/env python
"""
File: metrics
Date: 11/16/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import os
import numpy as np
import sklearn

from evaluation import plotting

# Test error divide by zero histogram

metric_descriptions = {
    "Hamming loss": "The fraction of misclassified labels",
    "ranking loss": "The average fraction of reversely ordered label pairs of each instance",
    "one-error": "The fraction of instances whose most confident label is irrelevant",
    "coverage": "The number of more labels on average should include to cover all relevant labels",
    "average precision": "The average fraction of relevant labels ranked higher than one other relevant label",

    "macro-F1": "F-measure averaging on each label",
    "instance-F1": "F-measure averaging on each instance",
    "micro-F1": "F-measure averaging on the prediction matrix",

    "macro-AUC": "AUC averaging on each label. S_macro is the set of correctly ordered instance pairs on each label",
    "instance-AUC": "AUC averaging on each instance. S_instance is the set of correctly ordered label pairs on each instance",
    "micro-AUC": "AUC averaging on prediction matrix. Smicro is the set of correct quadruples."
}


def create_report(labels, y_score, y_pred, output_dir, print=True):

    metrics = collect_evaluation_metrics(labels, y_score, y_pred)
    if print:
        print_evaluation_metrics(metrics)

    images = plot_evaluation_metrics(labels, y_score, y_pred, output_dir)

    report_file = os.path.join(output_dir, "report_metrics.md")
    save_metrics_report(metrics, report_file, images=images)


def collect_evaluation_metrics(labels, y_probs, y_pred):
    metrics = dict()

    metrics['Hamming loss'] = sklearn.metrics.hamming_loss(labels, y_pred)
    metrics['ranking loss'] = sklearn.metrics.label_ranking_loss(labels, y_probs)
    metrics['coverage'] = sklearn.metrics.coverage_error(labels, y_pred)
    metrics['average precision'] = sklearn.metrics.average_precision_score(labels, y_pred)

    metrics['macro-F1'] = sklearn.metrics.f1_score(labels, y_pred, average='macro')
    metrics['instance-F1'] = sklearn.metrics.f1_score(labels, y_pred, average='samples')
    metrics['micro-F1'] = sklearn.metrics.f1_score(labels, y_pred, average='micro')
    metrics['weighted-F1'] = sklearn.metrics.f1_score(labels, y_pred, average='weighted')

    metrics['macro-AUC'] = sklearn.metrics.roc_auc_score(labels, y_probs, average='macro')
    metrics['instance-AUC'] = sklearn.metrics.roc_auc_score(labels, y_probs, average='samples')
    metrics['micro-AUC'] = sklearn.metrics.roc_auc_score(labels, y_probs, average='micro')

    return metrics


def plot_evaluation_metrics(labels, y_score, y_pred, output_dir):
    assert isinstance(labels, np.ndarray)
    assert isinstance(y_score, np.ndarray)
    assert labels.shape == y_score.shape

    # list of images which were created
    image_files = list()

    # Plot cells counts with a given protein count
    label_hist_file = os.path.join(output_dir, "proteins_cell_counts.png")
    plotting.plot_label_histogram(labels, y_pred, save_file=label_hist_file)
    image_files.append(label_hist_file)

    # Plot counts of each type of protein in prediction set
    per_class_counts_file = os.path.join(output_dir, "proteins_label_counts.png")
    plotting.plot_num_each_protein(labels, y_pred, save_file=per_class_counts_file)
    image_files.append(per_class_counts_file)

    # plot per-class metrics
    per_class_images = plotting.plot_per_class_metrics(labels, y_score, y_pred)
    image_files.extend(per_class_images)
    return image_files


def print_evaluation_metrics(metrics):
    for metric, value in metrics.items():
        print("metric:\t%s" % value)


def save_metrics_report(metrics, output_file, images=None):
    assert isinstance(metrics, dict)

    report_lines = list()

    for metric, value in metrics.items():
        report_lines.append("%s: %s" % (metric, value))

    report_lines.append("---")
    # add in the images
    for image_file in images:
        line = "![%s](%s) " % (image_file, image_file)
        report_lines.append(line)
        
    report = "\n".join(report_lines)

    # save the report to file
    with open(output_file, "w+") as file:
        file.write(report)
