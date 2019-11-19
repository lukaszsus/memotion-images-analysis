import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

from settings import DATA_PATH, PROJECT_PATH
from visualization.utils import create_results_dir

sns.set()
sns.set_palette("pastel")


def make_scatter_pairplot(df: pd.DataFrame, plot_path, vars=None):
    plt.figure(figsize=(25, 25))
    if vars is None:
        ax = sns.pairplot(df, hue="label")
    else:
        ax = sns.pairplot(df, hue="label", vars=vars)
    plt.savefig(plot_path, bbox_inches='tight')


def plot_and_save(df, dataset_name, vars=None):
    # plot
    plot_path = os.path.join(PROJECT_PATH, "results/plots")
    plot_path = os.path.join(plot_path, dataset_name + "_scatter_plot.pdf")
    make_scatter_pairplot(df[df.columns[1:]], plot_path, vars)


def main(dataset_name):
    # paths
    create_results_dir()
    file_path = os.path.join(DATA_PATH, dataset_name + ".csv")
    labels_path = os.path.join(DATA_PATH, dataset_name + "_labels.pickle")

    # load data
    df = pd.read_csv(file_path)
    df["label_index"] = df["label_index"].astype(int)
    with open(labels_path, 'rb') as file:
        avail_labels = pickle.load(file)
    labels = list(map(lambda x: avail_labels[x], df["label_index"]))
    df["label"] = labels
    features = df.columns[:-1]
    df.columns = list(range(len(features))) + [df.columns[-1]]

    # plot
    plot_and_save(df, dataset_name)

    # index to feature name
    for i in range(len(features)):
        print("{}: {}".format(i, features[i]))


def main_per6(dataset_name):
    # paths
    create_results_dir()
    file_path = os.path.join(DATA_PATH, dataset_name + ".csv")
    labels_path = os.path.join(DATA_PATH, dataset_name + "_labels.pickle")

    # load data
    df = pd.read_csv(file_path)
    df["label_index"] = df["label_index"].astype(int)
    with open(labels_path, 'rb') as file:
        avail_labels = pickle.load(file)
    labels = list(map(lambda x: avail_labels[x], df["label_index"]))
    df["label"] = labels
    features = df.columns[:-1]
    df.columns = list(range(len(features))) + [df.columns[-1]]

    for i in range(1, 18, 6):
        plot_and_save(df, dataset_name + "_" + str(i), vars=df.columns[i:i+6])

    # index to feature name
    for i in range(len(features)):
        print("{}: {}".format(i, features[i]))


if __name__ == '__main__':
    # main("test_dataset")
    # main("test_dataset_hsv_var")
    main_per6("test_dataset_hsv_var")