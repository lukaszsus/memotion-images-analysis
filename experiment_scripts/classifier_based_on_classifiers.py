import os
import glob
import pickle as pkl
import re

import numpy as np

from classifiers.base_classifier import BaseClassifier
from classifiers.decision_trees import DecisionTree
from classifiers.neural_network import NeuralNetwork
from settings import DATA_PATH

ALL_CLFS = ['Gaussian Naive Bayes', 'Decision Tree (CART)', 'Random Forest',
            'k Nearest Neighbours', 'Multilayer Perceptron']


def load_classifiers_metrics(subdir='pics'):
    """
    Loads classifier metrics (y values for different classifiers) from given directory.
    :param subdir: directory name in *results/metrics* directory
    :return:
    """
    path = os.path.join(DATA_PATH, "results", "metrics", subdir)
    filenames = glob.glob(f'{path}/metrics_full_for-*')

    # filter for filenames that has full metrics (for all pictures)
    filenames[:] = [filename for filename in filenames if re.search('\d{3}', filename[-10:-7]) is None]

    clfs_y, methods_names = [], []
    for file_path in filenames:
        with open(file_path, "rb") as f:
            df_metrics, df_y, confusion_matrices = pkl.load(f)
            # print(df_metrics)
            clfs_y.append(df_y)
            methods_names.append(file_path.split('/')[-1].split('.')[0][17:])
            # print(df_y.shape)

    true_y = clfs_y[0]['True labels'].values
    return clfs_y, methods_names, true_y


def get_x_y(dataframes, y_true, clfs=ALL_CLFS):
    y_stacked = y_true.reshape(-1, 1)
    for df in dataframes:
        data = df[clfs]
        y_stacked = np.hstack((y_stacked, data.values))
    return y_stacked[:, 1:], y_stacked[:, 0]


def classifier_of_classifiers(sub_dir, y_labels, verbose=True):
    clfs_y, methods_names, y = load_classifiers_metrics(subdir=sub_dir)
    x, y = get_x_y(clfs_y, y)

    rf = DecisionTree(x, y, [], y_labels, max_depth=4, min_samples_leaf=10)
    y_pred = rf.crossval_random_forest(111)
    if verbose:
        rf.plot_confusion_matrix(y_pred)
        rf.show_basic_metrics(y_pred)


def majority_voting(sub_dir, y_labels, verbose=True):
    clfs_y, methods_names, y = load_classifiers_metrics(subdir=sub_dir)
    x, y = get_x_y(clfs_y, y)
    # print(x.shape)
    # print(x[0])
    # print(np.unique(x[0], return_counts=True))

    y_pred = []
    for i in range(x.shape[0]):
        labels, counts = np.unique(x[i], return_counts=True)
        l = [x for x, y in zip(labels, counts) if y == max(counts)][0]
        y_pred.append(l)

    if verbose:
        bc = BaseClassifier(x, y, [], y_labels)
        bc.plot_confusion_matrix(y_pred)
        bc.show_basic_metrics(y_pred)


if __name__ == "__main__":
    y_labels = ['cartoon', 'painting', 'photo', 'text']

    classifier_of_classifiers('memes_stand', y_labels)
    majority_voting('memes_stand', y_labels)




