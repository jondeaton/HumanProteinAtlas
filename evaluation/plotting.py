#!/usr/bin/env python
"""
File: plotting
Date: 12/4/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import sklearn
import numpy as np

import scikitplot as skplt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from HumanProteinAtlas import organelle_name


def plot_label_histogram(labels, pred_labels, save_file="proteins_cell_counts.png"):
    m = len(labels)

    true_counts_per_cell = [sum(labels[i]) for i in range(m)]
    pred_counts_per_cell = [sum(pred_labels[i]) for i in range(m)]

    max_count = max([max(true_counts_per_cell), max(pred_counts_per_cell)])
    count_range = np.arange(max_count + 1)

    true_cells_per_count = np.zeros(max_count + 1)
    pred_cells_per_count = np.zeros(max_count + 1)
    for i in range(m):
        true_cells_per_count[true_counts_per_cell[i]] += 1
        pred_cells_per_count[pred_counts_per_cell[i]] += 1

    width = 0.3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Text below each barplot
    plt.xticks([x + width / 2  for x in count_range], count_range)

    ax.bar(count_range, true_cells_per_count, width=width, color="g", label="True")
    ax.bar(count_range + width, pred_cells_per_count, width=width, color="r", label="Pred")

    plt.legend()
    plt.title("Number of cells vs. proteins counts in cell")
    plt.xlabel("Number of proteins in cell")
    plt.ylabel("Number of cells")
    plt.savefig(save_file)


def plot_num_each_protein(labels, pred_labels, save_file="proteins_label_counts.png"):
    m, n = labels.shape

    protein_names = organelle_name.values()
    assert len(protein_names) == n

    count_range = np.arange(n)
    true_protein_counts = [sum(labels[:, i]) for i in range(n)]
    pred_protein_counts = [sum(pred_labels[:, i]) for i in range(n)]

    width = 0.3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Text below each barplot
    plt.xticks([x + width / 2  for x in count_range], protein_names, rotation=90)

    ax.bar(count_range, true_protein_counts, width=width, color="g", label="True")
    ax.bar(count_range + width, pred_protein_counts, width=width, color="r", label="Pred")

    plt.legend()
    plt.title("Count of labels per protein")
    plt.xlabel("Protein")
    plt.ylabel("Label count")
    plt.savefig(save_file)


def plot_per_class_metrics(labels, y_probs, y_pred, output_dir):

    image_files = list() # save the list of image files created during this function

    accuracy_image = os.path.join(output_dir, "per-class_accuracy.png")
    plot_per_class_metric(labels, y_pred, sklearn.metrics.accuracy_score, "accuracy", save_file=accuracy_image)
    image_files.append(accuracy_image)

    precision_image = os.path.join(output_dir, "per-class_precision.png")
    plot_per_class_metric(labels, y_pred, sklearn.metrics.precision_score, "precision", save_file=precision_image)
    image_files.append(precision_image)

    recall_image = os.path.join(output_dir, "pre-class_recall.png")
    plot_per_class_metric(labels, y_pred, sklearn.metrics.recall_score, "recall", save_file=recall_image)
    image_files.append(recall_image)

    f1_image = os.path.join(output_dir, "per-class_f1.png")
    plot_per_class_metric(labels, y_pred, sklearn.metrics.f1_score, "F1", save_file=f1_image)
    image_files.append(f1_image)

    roc_auc_image = os.path.join(output_dir, "per-class_roc_auc.png")
    plot_per_class_metric(labels, y_probs, sklearn.metrics.roc_auc_score, "ROC AUC")
    image_files.append(roc_auc_image)

    # Per-class ROC
    roc_curves_image = os.path.join(output_dir, "roc_curves.png")
    skplt.metrics.plot_roc_curve(labels, y_probs)
    plt.savefig(roc_curves_image)
    image_files.append(roc_auc_image)

    # Per-class Precision Recall
    pr_curve_image = os.path.join(output_dir, "pr_curves.png")
    skplt.metrics.plot_precision_recall_curve(labels, y_probs)
    plt.savefig(pr_curve_image)
    image_files.append(pr_curve_image)

    return image_files


def plot_per_class_metric(labels, y_out, get_metric, metric_name, save_file=None):
    assert isinstance(labels, np.ndarray)

    m, c = labels.shape

    class_values = list()
    for i in range(c):
        value = get_metric(labels[:, i], y_out[:, i])
        class_values.append(value)

    plt.bar(class_values)
    plt.title("Per-class %s" % metric_name)
    plt.xlabel("Class")
    plt.ylabel(metric_name)

    if save_file is None:
        save_file = "per-class_%s.png" % metric_name
    plt.savefig(save_file)


# def plot_histogram(counts_matrix, x_tick_labels, legend_labels, title, xlabel, ylabel, save_file="histogram.png"):
#     m, n = counts_matrix.shape
#     assert len(x_tick_labels) == n
#
#     count_range = np.arange(n)
#
#     width = 0.3
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#
#     # Text below each barplot
#     plt.xticks([x + width / m for x in count_range], x_tick_labels, rotation=90)
#
#     colors = cm.rainbow(np.linspace(0, 1, m))
#
#     # Each bar type
#     for i in range(m):
#         ax.bar(count_range + i * width, counts_matrix[i], width=width, color=colors[i], label=legend_labels[i])
#
#     plt.legend()
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.show()
#     fig.savefig(outfile)
