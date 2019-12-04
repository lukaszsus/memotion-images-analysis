import os
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from classifiers.decision_trees import DecisionTree
from classifiers.k_nearest_neighbours import NearestNeighbours
from classifiers.naive_bayes import NaiveBayes
from classifiers.neural_network import NeuralNetwork
from settings import PROJECT_PATH


def load_dataset(filename):
    dataset_path = os.path.join(PROJECT_PATH, f'datasets_pkl/{filename}.pkl')
    with open(dataset_path, "rb") as f:
        x, y, x_feature_names, y_feature_names = pkl.load(f)
    return x, y, x_feature_names, y_feature_names


def decission_tree_metrics(x, y, x_labels, y_labels, max_depth=3, min_samples_leaf=5, estimators=15):
    dt = DecisionTree(x, y, x_labels, y_labels, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    label_tree = 'Decision Tree (CART)'
    y_pred_tree = dt.crossval_decision_tree()
    dt.plot_confusion_matrix(y_pred_tree, label=label_tree)
    dt.show_basic_metrics(y_pred_tree, label=label_tree)
    tree_acc, tree_f1 = dt.count_basic_metrics(y_pred_tree)

    label_forest = 'Random Forest'
    y_pred_forest = dt.crossval_random_forest(num_estimators=estimators)
    dt.plot_confusion_matrix(y_pred_forest, label=label_forest)
    dt.show_basic_metrics(y_pred_forest, label=label_forest)
    for_acc, for_f1 = dt.count_basic_metrics(y_pred_forest)

    return [[label_tree, tree_acc, tree_f1], [label_forest, for_acc, for_f1]]


def naive_bayes_metrics(x, y, x_labels, y_labels):
    nb = NaiveBayes(x, y, x_labels, y_labels)

    label = 'Gaussian Naive Bayes'
    y_pred = nb.crossval_gaussian_navie_bayes()
    nb.show_basic_metrics(y_pred, label=label)
    nb.plot_confusion_matrix(y_pred, label=label)
    acc, f1 = nb.count_basic_metrics(y_pred)
    return [[label, acc, f1]]


def knn_metrics(x, y, x_labels, y_labels, n_neighbours=3):
    nn = NearestNeighbours(x, y, x_labels, y_labels, n_neigh=n_neighbours)
    label = 'k Nearest Neighbours'
    y_pred_knn = nn.crossval_knn()
    nn.plot_confusion_matrix(y_pred_knn, label=label)
    nn.show_basic_metrics(y_pred_knn, label=label)
    acc, f1 = nn.count_basic_metrics(y_pred_knn)
    return [[label, acc, f1]]


def sklearn_mlp_metrics(x, y, x_labels, y_labels, hn=(256, ), activation='relu'):
    nn = NeuralNetwork(x, y, x_labels, y_labels, hidden_neurons=hn, activation_fun=activation)
    label = 'Multilayer Perceptron'
    y_pred_mlp = nn.crossval_mlp()
    nn.plot_confusion_matrix(y_pred_mlp, label=label)
    nn.show_basic_metrics(y_pred_mlp, label=label)
    acc, f1 = nn.count_basic_metrics(y_pred_mlp)
    return [[label, acc, f1]]


def plot_table_of_metrics(classifiers_metrics):
    table = pd.DataFrame(np.array(classifiers_metrics), columns=['Classfier', 'Accuracy', 'F-score'])
    plt.figure(figsize=(7, 2))

    cell_text = []
    for row in range(len(table)):
        cell_text.append(table.iloc[row])

    plt.table(cellText=cell_text, colLabels=table.columns, loc='center')
    plt.axis('off')
    plt.show()


def run_all_classifiers(x, y, x_labels, y_labels,
                        max_depth=3, min_samples_leaf=5, estimators=15, n_neighbours=3, hn=(256, ), activation='relu'):
    """
    Method that runs all classifiers and returns numpy array with accuracy and f-score for each (with classifier name).
    :param x: features values read from file
    :param y: label values (from 0 to 3) read from file
    :param x_labels: feature names
    :param y_labels: label names - cartoon, painting, photo and text
    :param max_depth: CART clf - maximum tree depth
    :param min_samples_leaf: CART clf - minimum number of samples in one leaf
    :param estimators: RandomForest clf - number of weak classifiers/trees
    :param n_neighbours: kNN clf - number of nearest neighbours
    :param hn: MLP clf - number of hidden layers (as a tuple)
    :param activation: MLP clf - activation function
    :return: numpy array of (clf_name, accuracy, f-score)
    """

    naive_bayes = naive_bayes_metrics(x, y, x_labels, y_labels)
    trees = decission_tree_metrics(x, y, x_labels, y_labels, max_depth, min_samples_leaf, estimators)
    knn = knn_metrics(x, y, x_labels, y_labels, n_neighbours)
    nn = sklearn_mlp_metrics(x, y, x_labels, y_labels, hn, activation)
    classifiers_metrics = np.array(naive_bayes + trees + knn + nn)
    classifiers_metrics[:, 1:] = np.around(np.array(classifiers_metrics[:, 1:], dtype=np.float32) * 100, 2)
    return classifiers_metrics


def load_multiple_datasets(filenames):
    x_features, x_names = [], []
    for filename in filenames:
        x, y, x_labels, y_labels = load_dataset(filename)
        x_features.append(np.array(x))
        x_names += [f'{x_label}_{filename}' for x_label in x_labels]

    x_features = np.reshape(x_features, (-1, len(x_names)))
    df = pd.DataFrame(np.concatenate((np.array(x_features), np.array(y).reshape(-1, 1)), axis=1),
                      columns=x_names+['label'])

    return df, y_labels


def load_x_y_from_files(filenames):
    """
    Method loading data from multiple .pkl files - changing dataframe to x and y (suitable for classifiers).
    :param filenames: list of filenames from datasets_pkl directory to load
    :return: array of x values concatenated for all files, array of y labels and feature names
    """
    df, y_labels = load_multiple_datasets(filenames)
    vals = df.values
    x_labels = df.columns[:-1]
    return vals[:, :-1], np.array(vals[:, -1], dtype=np.int32), x_labels, y_labels


if __name__ == "__main__":
    filenames = ['pics_bilateral_30_50']  # ['pics_bilateral_30_50', 'pics_bilateral_50_40']
    x, y, x_labels, y_labels = load_x_y_from_files(filenames)

    classifiers_metrics = run_all_classifiers(x, y, x_labels, y_labels)
    plot_table_of_metrics(classifiers_metrics)
