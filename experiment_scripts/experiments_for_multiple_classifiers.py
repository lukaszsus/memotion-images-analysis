import os
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from classifiers.decision_trees import DecisionTree
from classifiers.k_nearest_neighbours import NearestNeighbours
from classifiers.naive_bayes import NaiveBayes
from classifiers.neural_network import NeuralNetwork
from settings import PROJECT_PATH, DATA_PATH
from sklearn import preprocessing


def load_dataset(dirname, filename):
    dataset_path = os.path.join(DATA_PATH, f'{dirname}/{filename}.pickle')
    with open(dataset_path, "rb") as f:
        x = pkl.load(f)
    return x, [filename] * x.shape[1]


def decission_tree_metrics(x, y, x_labels, y_labels, one_vs_rest=False, max_depth=3, min_samples_leaf=5, estimators=15):
    dt = DecisionTree(x, y, x_labels, y_labels, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    label_tree = 'Decision Tree (CART)'
    y_pred_tree = dt.crossval_decision_tree(one_vs_rest)
    cm_tree = dt.plot_confusion_matrix(y_pred_tree, label=label_tree)
    dt.show_basic_metrics(y_pred_tree, label=label_tree)
    tree_acc, tree_f1 = dt.count_basic_metrics(y_pred_tree)

    label_forest = 'Random Forest'
    y_pred_forest = dt.crossval_random_forest(num_estimators=estimators, one_vs_rest=one_vs_rest)
    cm_forest = dt.plot_confusion_matrix(y_pred_forest, label=label_forest)
    dt.show_basic_metrics(y_pred_forest, label=label_forest)
    for_acc, for_f1 = dt.count_basic_metrics(y_pred_forest)

    return [[label_tree, tree_acc, tree_f1, y_pred_tree, cm_tree],
            [label_forest, for_acc, for_f1, y_pred_forest, cm_forest]]


def naive_bayes_metrics(x, y, x_labels, y_labels, one_vs_rest=False):
    nb = NaiveBayes(x, y, x_labels, y_labels)

    label = 'Gaussian Naive Bayes'
    y_pred = nb.crossval_gaussian_navie_bayes(one_vs_rest)
    nb.show_basic_metrics(y_pred, label=label)
    cm = nb.plot_confusion_matrix(y_pred, label=label)
    acc, f1 = nb.count_basic_metrics(y_pred)
    return [[label, acc, f1, y_pred, cm]]


def knn_metrics(x, y, x_labels, y_labels, one_vs_rest=False, n_neighbours=3):
    nn = NearestNeighbours(x, y, x_labels, y_labels, n_neigh=n_neighbours)
    label = 'k Nearest Neighbours'
    y_pred_knn = nn.crossval_knn(one_vs_rest)
    cm = nn.plot_confusion_matrix(y_pred_knn, label=label)
    nn.show_basic_metrics(y_pred_knn, label=label)
    acc, f1 = nn.count_basic_metrics(y_pred_knn)
    return [[label, acc, f1, y_pred_knn, cm]]


def sklearn_mlp_metrics(x, y, x_labels, y_labels, one_vs_rest=False, hn=(64,), activation='relu'):
    nn = NeuralNetwork(x, y, x_labels, y_labels, hidden_neurons=hn, activation_fun=activation)
    label = 'Multilayer Perceptron'
    y_pred_mlp = nn.crossval_mlp(one_vs_rest)
    cm = nn.plot_confusion_matrix(y_pred_mlp, label=label)
    nn.show_basic_metrics(y_pred_mlp, label=label)
    acc, f1 = nn.count_basic_metrics(y_pred_mlp)
    return [[label, acc, f1, y_pred_mlp, cm]]


def plot_table_of_metrics(classifiers_metrics, dirname, feature_set_name):
    table = pd.DataFrame(np.array(classifiers_metrics), columns=['Classfier', 'Accuracy', 'F-score'])
    plt.figure(figsize=(7, 2))

    cell_text = []
    for row in range(len(table)):
        cell_text.append(table.iloc[row])

    plt.table(cellText=cell_text, colLabels=table.columns, loc='center')
    plt.axis('off')

    datasets_names = '-'.join(feature_set_name)

    path = os.path.join(DATA_PATH, f'results/plots/{dirname}')
    file_name = f'plot_for-{datasets_names}.png'
    file_path = os.path.join(path, file_name)
    plt.savefig(file_path, bbox_inches='tight')

    path = os.path.join(DATA_PATH, f'results/tables/{dirname}')
    file_name = f'table_for-{datasets_names}.csv'
    file_path = os.path.join(path, file_name)
    table.to_csv(file_path, index=False)


def run_all_classifiers(x, y, x_labels, y_labels, one_vs_rest: bool = False,
                        max_depth=3, min_samples_leaf=5, estimators=15, n_neighbours=3, hn=(256,), activation='relu'):
    """
    Method that runs all classifiers and returns numpy array with accuracy and f-score for each (with classifier name).
    :param x: features values read from file
    :param y: label values (from 0 to 3) read from file
    :param x_labels: feature names
    :param y_labels: label names - cartoon, painting, photo and text
    :param one_vs_rest: whether to use one vs rest classification
    :param max_depth: CART clf - maximum tree depth
    :param min_samples_leaf: CART clf - minimum number of samples in one leaf
    :param estimators: RandomForest clf - number of weak classifiers/trees
    :param n_neighbours: kNN clf - number of nearest neighbours
    :param hn: MLP clf - number of hidden layers (as a tuple)
    :param activation: MLP clf - activation function
    :return: numpy array of (clf_name, accuracy, f-score), y_predicted and confussion_matrix)
    """

    naive_bayes = naive_bayes_metrics(x, y, x_labels, y_labels, one_vs_rest=one_vs_rest)
    trees = decission_tree_metrics(x, y, x_labels, y_labels, one_vs_rest, max_depth, min_samples_leaf, estimators)
    knn = knn_metrics(x, y, x_labels, y_labels, one_vs_rest, n_neighbours)
    nn = sklearn_mlp_metrics(x, y, x_labels, y_labels, one_vs_rest, hn, activation)
    classifiers_metrics = np.array(naive_bayes + trees + knn + nn)

    metrics, y_preds, confusion_matrices = classifiers_metrics[:, :-2], \
                                           classifiers_metrics[:, [0, -2]], \
                                           classifiers_metrics[:, [0, -1]]
    metrics[:, 1:] = np.around(np.array(metrics[:, 1:], dtype=np.float32) * 100, 2)

    return metrics, y_preds, confusion_matrices


def load_multiple_datasets(dirname, filenames):
    x_features, x_names = [], []
    for filename in filenames:
        x, x_labels = load_dataset(dirname, filename)
        x_features.append(np.array(x))
        x_names += [f'{x_label}_{filename}' for x_label in x_labels]

    x_features = np.concatenate(x_features, axis=1)
    x_features = np.reshape(x_features, (-1, len(x_names)))
    df = pd.DataFrame(np.array(x_features), columns=x_names)

    return df


def load_x_y_from_files(dirname, filenames):
    df = load_multiple_datasets(dirname, filenames)
    vals = df.values
    x_labels = df.columns[:-1]
    return vals, x_labels


def load_y_from_file(dirname, im_type):
    labels_path = os.path.join(DATA_PATH, f'{dirname}/{im_type}_labels.pickle')
    with open(labels_path, "rb") as f:
        y = pkl.load(f)
    return y


def save_cm_to_file(cm: np.ndarray, file_path: str, labels):
    name = cm[0]
    cm = cm[1]

    cm = pd.DataFrame(cm, index=labels, columns=labels).round(decimals=2)
    cm.index.name = 'True labels'
    cm.columns.name = 'Predicted labels'
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(cm, cmap="Greens", annot=True, fmt='', ax=ax)
    ax.set_title("{}".format(name))
    plt.savefig(file_path)


def save_cms_to_files(confusion_matrices: list, im_type: str, feature_set_name: str, y_labels):
    for cm in confusion_matrices:
        path = os.path.join(DATA_PATH, f'results/plots/{im_type}')
        datasets_names = '-'.join(feature_set_name)
        file_name = f'cm_for-{cm[0]}-{datasets_names}.png'
        file_path = os.path.join(path, file_name)
        save_cm_to_file(cm, file_path, y_labels)


def save_metrics_to_file(classifiers_metrics, y, y_preds, confusion_matrices, dirname, feature_set_name):
    """
    Saves couple of metrics to default directory: data/results/.
    :param classifiers_metrics: array of metrics (accuracy and fscore)
    :param y: true y values for each picture
    :param y_preds: y predictions for each classifier
    :param confusion_matrices: confusion matrices for all classifiers
    :param feature_set_name: list with names of datasets that where used for counting metrics
    """
    df_metrics = pd.DataFrame(np.array(classifiers_metrics), columns=['Classifier', 'Accuracy', 'Fscore'])
    df_y = pd.DataFrame(np.hstack([np.array(y).reshape(-1, 1), np.stack(y_preds[:, 1]).T]),
                        columns=['True labels'] + list(y_preds[:, 0]))
    path = os.path.join(DATA_PATH, f'results/metrics/{dirname}')
    datasets_names = '-'.join(feature_set_name)
    file_name = f'metrics_full_for-{datasets_names}.pickle'

    with open(os.path.join(path, file_name), "wb") as fout:
        pkl.dump([df_metrics, df_y, confusion_matrices], fout, protocol=-1)


def format_filenames(filenames: list, im_type: str):
    filenames_formatted = list()
    for filename in filenames:
        filenames_formatted.append(filename.format(im_type))
    return filenames_formatted


if __name__ == "__main__":
    filenames_list = [
        # # ['{}_bilateral_filter_h_from_hsv_differences'],
        # # ['{}_bilateral_filter_mean_color_diffs'],
        # # ['{}_bilateral_filter_n_color_diff'],
        ['{}_bilateral_filter_h_from_hsv_differences',
         '{}_bilateral_filter_mean_color_diffs',
         '{}_bilateral_filter_n_color_diff'],
        ['{}_color_counter_norm_color_count',
         '{}_edges_detector_grayscale_edges_factor'],
        ['{}_hsv_analyser_hsv_var'],
        ['{}_hsv_analyser_saturation_distribution'],
        ['{}_hsv_analyser_sat_value_distribution'],
        ['{}_bilateral_filter_h_from_hsv_differences',
         '{}_bilateral_filter_mean_color_diffs',
         '{}_bilateral_filter_n_color_diff',
         '{}_color_counter_norm_color_count',
         '{}_edges_detector_grayscale_edges_factor'],
        ['{}_hsv_analyser_hsv_var',
         '{}_hsv_analyser_saturation_distribution',
         '{}_hsv_analyser_sat_value_distribution'],
        ['{}_bilateral_filter_h_from_hsv_differences',
         '{}_bilateral_filter_mean_color_diffs',
         '{}_bilateral_filter_n_color_diff',
         '{}_color_counter_norm_color_count',
         '{}_edges_detector_grayscale_edges_factor',
         '{}_hsv_analyser_hsv_var',
         '{}_hsv_analyser_saturation_distribution',
         '{}_hsv_analyser_sat_value_distribution'],
        # # ['{}_kmeans_segementator_hsv_differences_15',
        # #  '{}_kmeans_segementator_hsv_differences_25',
        # #  '{}_kmeans_segementator_hsv_differences_35',
        # #  '{}_kmeans_segementator_hsv_differences_45',
        # #  '{}_kmeans_segementator_hsv_differences_55'],
        # # ['{}_kmeans_segementator_mean_color_diffs_15',
        # #  '{}_kmeans_segementator_mean_color_diffs_25',
        # #  '{}_kmeans_segementator_mean_color_diffs_35',
        # #  '{}_kmeans_segementator_mean_color_diffs_45',
        # #  '{}_kmeans_segementator_mean_color_diffs_55'],
        ['{}_bilateral_filter_h_from_hsv_differences',
         '{}_bilateral_filter_mean_color_diffs',
         '{}_bilateral_filter_n_color_diff',
         '{}_color_counter_norm_color_count',
         '{}_edges_detector_grayscale_edges_factor',
         '{}_hsv_analyser_hsv_var',
         '{}_hsv_analyser_saturation_distribution',
         '{}_hsv_analyser_sat_value_distribution',
         '{}_kmeans_segementator_hsv_differences_15',
         '{}_kmeans_segementator_hsv_differences_25',
         '{}_kmeans_segementator_hsv_differences_35',
         '{}_kmeans_segementator_hsv_differences_45',
         '{}_kmeans_segementator_hsv_differences_55',
         '{}_kmeans_segementator_mean_color_diffs_15',
         '{}_kmeans_segementator_mean_color_diffs_25',
         '{}_kmeans_segementator_mean_color_diffs_35',
         '{}_kmeans_segementator_mean_color_diffs_45',
         '{}_kmeans_segementator_mean_color_diffs_55']
        # # ['{}_kmeans_segementator_hsv_differences_3',
        # #  '{}_kmeans_segementator_hsv_differences_6',
        # #  '{}_kmeans_segementator_hsv_differences_9',
        # #  '{}_kmeans_segementator_hsv_differences_12'],
        # # ['{}_kmeans_segementator_mean_color_diffs_3',
        # #  '{}_kmeans_segementator_mean_color_diffs_6',
        # #  '{}_kmeans_segementator_mean_color_diffs_9',
        # #  '{}_kmeans_segementator_mean_color_diffs_12'],
        # # ['{}_bilateral_filter_h_from_hsv_differences',
        # #  '{}_bilateral_filter_mean_color_diffs',
        # #  '{}_bilateral_filter_n_color_diff',
        # #  '{}_color_counter_norm_color_count',
        # #  '{}_edges_detector_grayscale_edges_factor',
        # #  '{}_hsv_analyser_hsv_var',
        # #  '{}_hsv_analyser_saturation_distribution',
        # #  '{}_hsv_analyser_sat_value_distribution',
        # #  '{}_kmeans_segementator_hsv_differences_3',
        # #  '{}_kmeans_segementator_hsv_differences_6',
        # #  '{}_kmeans_segementator_hsv_differences_9',
        # #  '{}_kmeans_segementator_hsv_differences_12',
        # #  '{}_kmeans_segementator_mean_color_diffs_3',
        # #  '{}_kmeans_segementator_mean_color_diffs_6',
        # #  '{}_kmeans_segementator_mean_color_diffs_9',
        # #  '{}_kmeans_segementator_mean_color_diffs_12'],
        # ['{}_gabor_filter'],
        # ['{}_bilateral_filter_h_from_hsv_differences',
        #  '{}_bilateral_filter_mean_color_diffs',
        #  '{}_bilateral_filter_n_color_diff',
        #  '{}_color_counter_norm_color_count',
        #  '{}_edges_detector_grayscale_edges_factor',
        #  '{}_gabor_filter'],
        # ['{}_bilateral_filter_h_from_hsv_differences',
        #  '{}_bilateral_filter_mean_color_diffs',
        #  '{}_bilateral_filter_n_color_diff',
        #  '{}_color_counter_norm_color_count',
        #  '{}_edges_detector_grayscale_edges_factor',
        #  '{}_hsv_analyser_hsv_var',
        #  '{}_hsv_analyser_saturation_distribution',
        #  '{}_hsv_analyser_sat_value_distribution',
        #  '{}_gabor_filter'],
        # ['{}_bilateral_filter_h_from_hsv_differences',
        #  '{}_bilateral_filter_mean_color_diffs',
        #  '{}_bilateral_filter_n_color_diff',
        #  '{}_color_counter_norm_color_count',
        #  '{}_edges_detector_grayscale_edges_factor',
        #  '{}_hsv_analyser_hsv_var',
        #  '{}_hsv_analyser_saturation_distribution',
        #  '{}_hsv_analyser_sat_value_distribution',
        #  '{}_kmeans_segementator_hsv_differences_15',
        #  '{}_kmeans_segementator_hsv_differences_25',
        #  '{}_kmeans_segementator_hsv_differences_35',
        #  '{}_kmeans_segementator_hsv_differences_45',
        #  '{}_kmeans_segementator_hsv_differences_55',
        #  '{}_kmeans_segementator_mean_color_diffs_15',
        #  '{}_kmeans_segementator_mean_color_diffs_25',
        #  '{}_kmeans_segementator_mean_color_diffs_35',
        #  '{}_kmeans_segementator_mean_color_diffs_45',
        #  '{}_kmeans_segementator_mean_color_diffs_55',
        #  '{}_gabor_filter'],
        # ['{}_bilateral_filter_h_from_hsv_differences',
        #  '{}_bilateral_filter_mean_color_diffs',
        #  '{}_bilateral_filter_n_color_diff',
        #  '{}_color_counter_norm_color_count',
        #  '{}_edges_detector_grayscale_edges_factor',
        #  '{}_kmeans_segementator_hsv_differences_15',
        #  '{}_kmeans_segementator_hsv_differences_25',
        #  '{}_kmeans_segementator_hsv_differences_35',
        #  '{}_kmeans_segementator_hsv_differences_45',
        #  '{}_kmeans_segementator_hsv_differences_55',
        #  '{}_gabor_filter']
        # ['{}_pca_top_5'],
        # ['{}_pca_top_10'],
        # ['{}_pca_top_15'],
        # ['{}_pca_top_25'],
        # ['{}_pca_top_50']
    ]

    features_names = [
        ['bilateral_filter_h_from_hsv_differences'],
        ['bilateral_filter_mean_color_diffs'],
        ['bilateral_filter_n_color_diff'],
        ['bilateral_filter'],
        ['color_counter_edges_detector'],
        ['hsv_analyser_hsv_var'],
        ['hsv_analyser_saturation_distribution'],
        ['hsv_analyser_sat_value_distribution'],
        ['scalar'],
        ['hsv_analyser'],
        ['scalar_hsv'],
        ['kmeans_hsv'],
        ['kmeans_mean'],
        ['scalar_hsv_kmeans'],
        # ['kmeans_hsv3'],
        # ['kmeans_mean3'],
        # ['scalar_hsv_kmeans3']
        # ['gabor_filter'],
        # ['scalar_gabor'],
        # ['scalar_hsv_gabor'],
        # ['scalar_hsv_kmeans_gabor'],
        # ['scalar_kmeans_gabor']
        # ['pca_top_5'],
        # ['pca_top_10'],
        # ['pca_top_15'],
        # ['pca_top_25'],
        # ['pca_top_50']
    ]

    im_types = ["pics", "memes"]
    stands = [True, False]
    y_labels = ["cartoon", "painting", "photo", "text"]
    one_vs_rests = [True]  # , False]
    for one_vs_rest in one_vs_rests:
        for im_type in im_types:
            for stand in stands:
                dirname = im_type + "_feature_binaries"
                y = load_y_from_file(dirname, im_type)
                for i in range(len(filenames_list)):
                    filenames = filenames_list[i]
                    feature_set_name = features_names[i]
                    filenames_formatted = format_filenames(filenames, im_type)
                    x, x_labels = load_x_y_from_files(dirname, filenames_formatted)

                    if stand:
                        im_type_stand = im_type + "_stand"
                        x = preprocessing.scale(x)
                    else:
                        im_type_stand = im_type

                    if one_vs_rest:
                        im_type_stand = im_type_stand + "_one_vs_rest"

                    classifiers_metrics, y_preds, confusion_matrices = \
                        run_all_classifiers(x, y, x_labels, y_labels, one_vs_rest)
                    plot_table_of_metrics(classifiers_metrics, im_type_stand, feature_set_name)
                    save_metrics_to_file(classifiers_metrics, y, y_preds, confusion_matrices, im_type_stand,
                                         feature_set_name)
                    save_cms_to_files(confusion_matrices, im_type_stand, feature_set_name, y_labels)
