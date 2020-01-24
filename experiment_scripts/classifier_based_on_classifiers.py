import os
import glob
import pickle as pkl
import re
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    # print(len(filenames) * 5)
    clfs_y, methods_names, cms = [], [], []
    for file_path in filenames:
        with open(file_path, "rb") as f:
            df_metrics, df_y, confusion_matrices = pkl.load(f)
            cms.append(confusion_matrices)
            clfs_y.append(df_y)
            methods_names.append(file_path.split('/')[-1].split('.')[0][17:])
            # print(df_y.shape)

    true_y = clfs_y[0]['True labels'].values
    return clfs_y, methods_names, true_y, cms


def get_x_y(dataframes, y_true, clfs=ALL_CLFS):
    y_stacked = y_true.reshape(-1, 1)
    for df in dataframes:
        data = df[clfs]
        y_stacked = np.hstack((y_stacked, data.values))
    return y_stacked[:, 1:], y_stacked[:, 0]


def classifier_of_classifiers(x, y, y_labels, no_trees=111, verbose=True):
    rf = DecisionTree(x, y, [], y_labels, max_depth=4, min_samples_leaf=10)
    y_pred = rf.crossval_random_forest(no_trees)
    if verbose:
        cm = rf.plot_confusion_matrix(y_pred)
        rf.show_basic_metrics(y_pred)
        return cm, y_pred
    return rf.count_basic_metrics(y_pred)


def majority_voting(x, y, y_labels, verbose=True):
    y_pred = []
    for i in range(x.shape[0]):
        labels, counts = np.unique(x[i], return_counts=True)
        l = [x for x, y in zip(labels, counts) if y == max(counts)][0]
        y_pred.append(l)

    if verbose:
        bc = BaseClassifier(x, y, [], y_labels)
        cm = bc.plot_confusion_matrix(y_pred)
        bc.show_basic_metrics(y_pred)
        return cm, y_pred


def plot_cm_pink_xD(cm, name):
    labels = ['cartoon', 'painting', 'photo', 'text']
    cm = pd.DataFrame(cm, index=labels, columns=labels).round(decimals=2)
    cm.index.name = 'True labels'
    cm.columns.name = 'Predicted labels'
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, cmap="PuRd", annot=True, fmt='', ax=ax)
    ax.set_title("{}\n".format(name))
    plt.show()


if __name__ == "__main__":
    y_labels = ['cartoon', 'painting', 'photo', 'text']
    subdir = 'pics_stand_one_vs_rest'  # 'pics_stand_one_vs_rest'

    clfs_y, methods_names, y, cms = load_classifiers_metrics(subdir=subdir)
    x, y = get_x_y(clfs_y, y)

    # cm_rf, y_pred_rf = classifier_of_classifiers(x, y, y_labels, no_trees=128)
    # cm_voting, y_pred_voting = majority_voting(x, y, y_labels)
    #
    # sns.set_context(rc={"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13})
    # plot_cm_pink_xD(cm_rf, 'Random Forest')
    # plot_cm_pink_xD(cm_voting, 'Majority Voting')

    # pd.set_option('display.max_rows', 150)
    # df = pd.DataFrame(np.array([y, y_pred_voting]).T, columns=['True labels', 'Clf'])
    # print(df[df['True labels'] != df['Clf']])

    sns.set()
    sns.set_palette('PuRd_r')

    trees = [8, 16, 32, 64, 128, 256, 512]
    accs, times = [], []
    for i in range(10):
        print(f'{i+1} round...')
        accs.append([])
        times.append([])
        for t in trees:
            start = time.time()
            acc, f1 = classifier_of_classifiers(x, y, y_labels, no_trees=t, verbose=False)
            end = time.time()
            accs[i].append(acc)
            times[i].append(end - start)

    plt.figure(figsize=(7, 4))
    plt.plot(trees, np.mean(accs, axis=0), linewidth=5)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.title('Wykres zależności dokładności predykcji (accuracy)\nod liczby drzew \n', fontsize=14)
    # plt.xscale('log')
    plt.show()

    sns.set_palette('PiYG')
    plt.figure(figsize=(7, 6))
    plt.plot(trees, np.mean(times, axis=0), linewidth=5)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.title('Wykres zależności czasu od liczby drzew \n', fontsize=14)
    # plt.xscale('log')
    plt.show()
