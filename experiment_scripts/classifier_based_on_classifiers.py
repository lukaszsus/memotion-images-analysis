import os
import glob
import pickle as pkl
import numpy as np

from classifiers.neural_network import NeuralNetwork
from settings import DATA_PATH

ALL_CLFS = ['Gaussian Naive Bayes', 'Decision Tree (CART)', 'Random Forest',
            'k Nearest Neighbours', 'Multilayer Perceptron']


def load_classifiers_metrics(type='pics'):
    path = os.path.join(DATA_PATH, "results", "metrics", type)
    filenames = glob.glob(f'{path}/metrics_full_for-*')

    clfs_y, methods_names = [], []
    for file_path in filenames:
        with open(file_path, "rb") as f:
            df_metrics, df_y, confusion_matrices = pkl.load(f)
            print(df_metrics)
            clfs_y.append(df_y)
            methods_names.append(file_path.split('/')[-1].split('.')[0][17:])

    true_y = clfs_y[0]['True labels'].values
    return clfs_y, methods_names, true_y


def get_x_y(dataframes, y_true, clfs=ALL_CLFS):
    y_stacked = y_true.reshape(-1, 1)
    for df in dataframes:
        data = df[clfs]
        y_stacked = np.hstack((y_stacked, data.values))
    return y_stacked[:, 1:], y_stacked[:, 0]


if __name__ == "__main__":
    clfs_y, methods_names, y = load_classifiers_metrics()
    x, y = get_x_y(clfs_y, y)

    nn = NeuralNetwork(x, y, [], [], hidden_neurons=())

    y_pred_mlp = nn.crossval_mlp()
    nn.plot_confusion_matrix(y_pred_mlp)
    nn.show_basic_metrics(y_pred_mlp)