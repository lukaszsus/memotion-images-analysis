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
    plt.figure(figsize=(6, 2))

    cell_text = []
    for row in range(len(table)):
        cell_text.append(table.iloc[row])

    plt.table(cellText=cell_text, colLabels=table.columns, loc='center')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    filename = 'pics_bilateral_50_40'
    x, y, x_labels, y_labels = load_dataset(filename)

    naive_bayes = naive_bayes_metrics(x, y, x_labels, y_labels)
    trees = decission_tree_metrics(x, y, x_labels, y_labels)
    knn = knn_metrics(x, y, x_labels, y_labels)
    nn = sklearn_mlp_metrics(x, y, x_labels, y_labels)
    classifiers_metrics = np.array(naive_bayes + trees + knn + nn)

    plot_table_of_metrics(classifiers_metrics)
