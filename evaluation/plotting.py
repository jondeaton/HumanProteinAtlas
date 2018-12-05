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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve

from HumanProteinAtlas import organelle_name


def plot_evaluation_metrics(labels, y_score, y_pred, output_dir):
    assert isinstance(labels, np.ndarray)
    assert isinstance(y_score, np.ndarray)
    assert labels.shape == y_score.shape

    # list of images which were created
    image_files = list()

    # plot per-class metrics
    per_class_images = plot_per_class_metrics(labels, y_score, y_pred, output_dir)
    image_files.extend(per_class_images)

    # Plot counts of each type of protein in prediction set
    per_class_counts_file = "proteins_label_counts.png"
    plot_num_each_protein(labels, y_pred, save_file=os.path.join(output_dir, per_class_counts_file))
    image_files.append(per_class_counts_file)

    return image_files


def plot_per_class_metrics(y_true, y_probs, y_pred, output_dir):

    image_files = list()  # save the list of image files created during this function

    accuracy_image = "per-class_accuracy.png"
    plot_per_class_metric(y_true, y_pred, sklearn.metrics.accuracy_score, "accuracy",
                          save_file=os.path.join(output_dir, accuracy_image))
    image_files.append(accuracy_image)

    precision_image = "per-class_precision.png"
    plot_per_class_metric(y_true, y_pred, sklearn.metrics.precision_score, "precision",
                          save_file=os.path.join(output_dir, precision_image))
    image_files.append(precision_image)

    recall_image = "pre-class_recall.png"
    plot_per_class_metric(y_true, y_pred, sklearn.metrics.recall_score, "recall",
                          save_file=os.path.join(output_dir, recall_image))
    image_files.append(recall_image)

    f1_image = "per-class_f1.png"
    plot_per_class_metric(y_true, y_pred, sklearn.metrics.f1_score, "F1",
                          save_file=os.path.join(output_dir, f1_image))
    image_files.append(f1_image)

    roc_auc_image = "per-class_roc_auc.png"
    plot_per_class_metric(y_true, y_probs, sklearn.metrics.roc_auc_score, "ROC AUC",
                          save_file=os.path.join(output_dir, roc_auc_image))
    image_files.append(roc_auc_image)

    # Per-class ROC
    roc_curves_file = "roc_curves.png"
    plot_roc(y_true, y_probs)
    plt.savefig(os.path.join(output_dir, roc_curves_file))
    image_files.append(roc_curves_file)

    # Per-class Precision Recall
    pr_curves_file = "pr_curves.png"
    plot_precision_recall(y_true, y_probs)
    plt.savefig(os.path.join(output_dir, pr_curves_file))
    image_files.append(pr_curves_file)

    return image_files


def plot_per_class_metric(labels, y_probs, get_metric, metric_name, save_file=None):
    assert isinstance(labels, np.ndarray)
    assert isinstance(y_probs, np.ndarray)

    class_values = list()
    m, c = labels.shape
    for i in range(c):
        value = get_metric(labels[:, i], y_probs[:, i])
        class_values.append(value)

    plt.figure()
    plt.bar(range(c), class_values)
    plt.title("Per-class %s" % metric_name)
    plt.xlabel("Class")
    plt.ylabel(metric_name)
    plt.ylim((0, 1))

    if save_file is None:
        save_file = "per-class_%s.png" % metric_name
    plt.savefig(save_file)


def plot_roc(y_true, y_probas, title='ROC Curves', plot_micro=True, plot_macro=True, classes_to_plot=None,
             ax=None, figsize=None, cmap='nipy_spectral', title_fontsize="large", text_fontsize=5):

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    m, num_classes = y_probas.shape
    classes = np.arange(num_classes)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    fpr_dict = dict()
    tpr_dict = dict()

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true[:, i], probas[:, i])
        if to_plot:
            roc_auc = auc(fpr_dict[i], tpr_dict[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr_dict[i], tpr_dict[i], lw=2, color=color,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                          ''.format(classes[i], roc_auc))

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return ax


def plot_precision_recall(y_true, y_probas, title='Precision-Recall Curve', plot_micro=True, classes_to_plot=None,
                          ax=None, figsize=None, cmap='nipy_spectral', title_fontsize=5, text_fontsize="medium"):


    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    m, num_classes = y_probas.shape
    classes = np.arange(num_classes)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    binarized_y_true = y_true

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        if to_plot:
            average_precision = average_precision_score(binarized_y_true[:, i], probas[:, i])
            precision, recall, _ = precision_recall_curve(y_true[:, i], probas[:, i], pos_label=classes[i])

            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(recall, precision, lw=2,
                    label='Precision-recall curve of class {0} '
                          '(area = {1:0.3f})'.format(classes[i],
                                                     average_precision),
                    color=color)

    if plot_micro:
        precision, recall, _ = precision_recall_curve(
            binarized_y_true.ravel(), probas.ravel())
        average_precision = average_precision_score(binarized_y_true,
                                                    probas,
                                                    average='micro')
        ax.plot(recall, precision,
                label='micro-average Precision-recall curve '
                      '(area = {0:0.3f})'.format(average_precision),
                color='navy', linestyle=':', linewidth=4)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(labelsize=text_fontsize)
    # ax.legend(loc='best', fontsize=text_fontsize)
    return ax


def plot_label_histogram(labels, pred_labels, save_file="proteins_cell_counts.png"):
    assert isinstance(labels, np.ndarray)
    assert isinstance(pred_labels, np.ndarray)
    m, num_classes = labels.shape

    true_counts_per_cell = labels.sum(axis=0).astype(int)
    pred_counts_per_cell = pred_labels.sum(axis=0).astype(int)

    max_count = max(true_counts_per_cell.max(), pred_counts_per_cell.max())
    count_range = np.arange(max_count + 1)

    true_cells_per_count = np.zeros(max_count + 1)
    pred_cells_per_count = np.zeros(max_count + 1)
    for i in range(num_classes):
        true_cells_per_count[true_counts_per_cell[i]] += 1
        pred_cells_per_count[pred_counts_per_cell[i]] += 1

    width = 0.3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Text below each barplot
    plt.xticks([x + width / 2 for x in count_range], count_range)

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
