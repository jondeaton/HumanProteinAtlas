#!/usr/bin/env python
"""
File: metrics
Date: 11/16/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import os
import numpy as np
import sklearn
import evaluation.plotting

# Test error divide by zero histogram

metric_descriptions = {
    "macro-F1": "F-measure averaging on each label",
    "instance-F1": "F-measure averaging on each instance",
    "micro-F1": "F-measure averaging on the prediction matrix",
    "weighted-F1": "F-measure calculated for each label, and averaged as weighted by the number of true instances for each label",

    "Hamming loss": "The fraction of misclassified labels",
    "ranking loss": "The average fraction of reversely ordered label pairs of each instance",
    "one-error": "The fraction of instances whose most confident label is irrelevant",
    "coverage": "The number of more labels on average should include to cover all relevant labels",
    "average precision": "The average fraction of relevant labels ranked higher than one other relevant label",

    "macro-AUC": "AUC averaging on each label. S_macro is the set of correctly ordered instance pairs on each label",
    "instance-AUC": "AUC averaging on each instance. S_instance is the set of correctly ordered label pairs on each instance",
    "micro-AUC": "AUC averaging on prediction matrix. Smicro is the set of correct quadruples."
}


def create_report(y_true, y_score, y_pred, output_dir, print=True):
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_score, np.ndarray)
    assert isinstance(y_pred, np.ndarray)

    metrics = collect_evaluation_metrics(y_true, y_score, y_pred)
    if print:
        print_evaluation_metrics(metrics)

    images = evaluation.plotting.plot_evaluation_metrics(y_true, y_score, y_pred, output_dir)

    report_file = os.path.join(output_dir, "report_metrics.md")
    save_metrics_report(metrics, report_file, images=images)


def collect_evaluation_metrics(y_true, y_probs, y_pred):
    metrics = dict()

    metrics['macro-F1'] = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    metrics['instance-F1'] = sklearn.metrics.f1_score(y_true, y_pred, average='samples')
    metrics['micro-F1'] = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
    metrics['weighted-F1'] = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')

    metrics['Hamming loss'] = sklearn.metrics.hamming_loss(y_true, y_pred)
    metrics['ranking loss'] = sklearn.metrics.label_ranking_loss(y_true, y_probs)
    metrics['one-error'] = sklearn.metrics.zero_one_loss(y_true, y_pred)
    metrics['coverage'] = sklearn.metrics.coverage_error(y_true, y_pred)
    metrics['average precision'] = sklearn.metrics.average_precision_score(y_true, y_pred)

    metrics['macro-AUC'] = sklearn.metrics.roc_auc_score(y_true, y_probs, average='macro')
    metrics['instance-AUC'] = sklearn.metrics.roc_auc_score(y_true, y_probs, average='samples')
    metrics['micro-AUC'] = sklearn.metrics.roc_auc_score(y_true, y_probs, average='micro')

    return metrics


def print_evaluation_metrics(metrics):
    for metric, value in metrics.items():
        print("%s:\t%s" % (metric, value))


def save_metrics_report(metrics, output_file, images=None):
    assert isinstance(metrics, dict)

    report_lines = list()

    # Make a nice table
    report_lines.append("| Metric | Value | Description |")
    report_lines.append("|---|---|---|")

    for metric, value in metrics.items():
        report_lines.append("| %s | %s | %s |" % (metric, value, metric_descriptions[metric]))

    report_lines.append("------")

    # add in the images
    for image_file in images:
        image_filename = os.path.basename(image_file)
        line = "![%s](%s) " % (image_filename, image_filename)
        report_lines.append(line)

    report = "\n".join(report_lines)

    # save the report to file
    with open(output_file, "w+") as file:
        file.write(report)
