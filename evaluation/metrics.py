#!/usr/bin/env python
"""
File: metrics
Date: 11/16/18 
Author: Robert Neff (rneff@stanford.edu)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn import metrics

from HumanProteinAtlas import organelle_name


# Test error divide by zero histogram


def evaluation_metrics(labels, y_probs, output_file="metrics.txt"):
    assert isinstance(labels, np.ndarray)
    assert isinstance(y_probs, np.ndarray)
    assert labels.shape == y_probs.shape

    pred_labels = y_probs.copy()
    pred_labels[pred_labels > 0.5] = 1
    pred_labels[pred_labels < 0.5] = 0

    with open(output_file, "w+") as file:
        # F-measure averaging on each label
        f1_macro_score = metrics.f1_score(labels, pred_labels, average='macro')
        print_and_write_line(file, "F1-macro score: " + str(f1_macro_score))

        # Fraction of misclassified labels
        hamming_loss = metrics.hamming_loss(labels, pred_labels)
        print_and_write_line(file, "Hamming loss: " + str(hamming_loss))

        # The average fraction of relevant labels ranked
        # higher than one other relevant label
        avg_precision_score = metrics.average_precision_score(labels, pred_labels)
        print_and_write_line(file, "Average precision score: " + str(avg_precision_score))

        # The number of more labels on average should
        # include to cover all relevant labels
        coverage = metrics.coverage_error(labels, pred_labels)
        print_and_write_line(file, "Coverage error: " + str(coverage))

        # Plot cells counts with a given protein count
        plot_label_histogram(labels, pred_labels)

        # Plot counts of each type of protein in prediction set
        plot_num_each_protein(labels, pred_labels)


def plot_label_histogram(labels, pred_labels):
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
    fig.savefig("outputs/proteins_cell_counts.png")


def plot_num_each_protein(labels, pred_labels):
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
    fig.savefig("outputs/proteins_label_counts.png")


def print_and_write_line(file, str):
    print(str)
    file.write(str + "\n")
