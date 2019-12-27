import os
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn import preprocessing

from experiment_scripts.experiments_for_multiple_classifiers import load_y_from_file, format_filenames, \
    load_x_y_from_files, plot_table_of_metrics, save_metrics_to_file, save_cms_to_files, run_all_classifiers
from settings import DATA_PATH


def save_x_to_file(dirname, im_type, filename, x):
    dataset_path = os.path.join(DATA_PATH, f'{dirname}/{im_type}_{filename}.pickle')
    with open(dataset_path, 'wb') as f:
        pkl.dump(x, f)


def save_y_to_file(dirname, im_type, y):
    dataset_path = os.path.join(DATA_PATH, f'{dirname}/{im_type}_labels.pickle')
    with open(dataset_path, 'wb') as f:
        pkl.dump(y, f)


def save_size_metrics_to_file(dirname, df_acc, df_f1, filename):
    path = os.path.join(DATA_PATH, f'results/tables/{dirname}')
    file_path = os.path.join(path, f'accuracy-{filename}.csv')
    df_acc.to_csv(file_path, index=False)

    file_path = os.path.join(path, f'f1score-{filename}.csv')
    df_f1.to_csv(file_path, index=False)


def shuffle_x_y(x, y):
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return x[p], y[p]


# def load_labels_from_file(dirname):
#     labels_path = os.path.join(DATA_PATH, f'{dirname}/labels.pickle')
#     with open(labels_path, "rb") as f:
#         y = pkl.load(f)
#     return y


def create_combined_features(filenames_list, features_names, im_types):
    out_im_type = "combined"
    for i in range(len(filenames_list)):
        x_list = list()
        y_list = list()

        for im_type in im_types:
            dirname = im_type + "_feature_binaries"
            y = load_y_from_file(dirname, im_type)
            filenames = filenames_list[i]
            filenames_formatted = format_filenames(filenames, im_type)
            x, x_labels = load_x_y_from_files(dirname, filenames_formatted)

            if im_type == "pics":
                no_text = y != 3
                y = y[no_text]
                x = x[no_text]

            x_list.append(x)
            y_list.append(y)

        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        print(x.shape)
        print(y.shape)
        print()

        dirname = out_im_type + "_feature_binaries"
        feature_set_name = features_names[i][0]
        save_x_to_file(dirname, out_im_type, feature_set_name, x)
        save_y_to_file(dirname, out_im_type, y)


def experiments_on_dataset_size(features_names, dataset_sizes, stands, im_type, y_labels):
    dirname = im_type + "_feature_binaries"
    for i in range(len(features_names)):
        y_all = load_y_from_file(dirname, im_type)
        feature_set_name = features_names[i]
        x_all, x_labels = load_x_y_from_files(dirname, [im_type + "_" + feature_set_name[0]])

        accuracies = {"Dataset Size": list(),
                      "Gaussian Naive Bayes": list(),
                      "Decision Tree (CART)": list(),
                      "Random Forest": list(),
                      "k Nearest Neighbours": list(),
                      "Multilayer Perceptron": list()}

        f1_scores = {"Dataset Size": list(),
                     "Gaussian Naive Bayes": list(),
                     "Decision Tree (CART)": list(),
                     "Random Forest": list(),
                     "k Nearest Neighbours": list(),
                     "Multilayer Perceptron": list()}

        x_all, y_all = shuffle_x_y(x_all, y_all)

        for stand in stands:
            for ds_size in dataset_sizes:
                if ds_size is not None:
                    y = y_all[:ds_size]
                    x = x_all[:ds_size]
                else:
                    y = y_all
                    x = x_all

                if stand:
                    im_type_stand = im_type + "_stand"
                    x = preprocessing.scale(x)
                else:
                    im_type_stand = im_type

                classifiers_metrics, y_preds, confusion_matrices = run_all_classifiers(x, y, x_labels, y_labels)

                accuracies["Dataset Size"].append(ds_size)
                f1_scores["Dataset Size"].append(ds_size)
                for class_metric in classifiers_metrics:
                    accuracies[class_metric[0]].append(class_metric[1])
                    f1_scores[class_metric[0]].append(class_metric[2])

                plot_table_of_metrics(classifiers_metrics, im_type_stand, [feature_set_name[0] + "-" + str(ds_size)])
                save_metrics_to_file(classifiers_metrics, y, y_preds, confusion_matrices, im_type_stand,
                                     [feature_set_name[0] + "-" + str(ds_size)])
                save_cms_to_files(confusion_matrices, im_type_stand, [feature_set_name[0] + "-" + str(ds_size)],
                                  y_labels)

            df_acc = pd.DataFrame(accuracies)
            df_f1 = pd.DataFrame(f1_scores)

            save_size_metrics_to_file(im_type_stand, df_acc, df_f1, feature_set_name[0])


if __name__ == "__main__":
    filenames_list = [
        # ['{}_bilateral_filter_h_from_hsv_differences',
        #  '{}_bilateral_filter_mean_color_diffs',
        #  '{}_bilateral_filter_n_color_diff'],
        # ['{}_color_counter_norm_color_count',
        #  '{}_edges_detector_grayscale_edges_factor'],
        # ['{}_hsv_analyser_hsv_var'],
        # ['{}_hsv_analyser_saturation_distribution'],
        # ['{}_hsv_analyser_sat_value_distribution'],
        # ['{}_bilateral_filter_h_from_hsv_differences',
        #  '{}_bilateral_filter_mean_color_diffs',
        #  '{}_bilateral_filter_n_color_diff',
        #  '{}_color_counter_norm_color_count',
        #  '{}_edges_detector_grayscale_edges_factor'],
        # ['{}_hsv_analyser_hsv_var',
        #  '{}_hsv_analyser_saturation_distribution',
        #  '{}_hsv_analyser_sat_value_distribution'],
        # ['{}_bilateral_filter_h_from_hsv_differences',
        #  '{}_bilateral_filter_mean_color_diffs',
        #  '{}_bilateral_filter_n_color_diff',
        #  '{}_color_counter_norm_color_count',
        #  '{}_edges_detector_grayscale_edges_factor',
        #  '{}_hsv_analyser_hsv_var',
        #  '{}_hsv_analyser_saturation_distribution',
        #  '{}_hsv_analyser_sat_value_distribution'],
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
        #  '{}_kmeans_segementator_mean_color_diffs_55'],
        # ['{}_bilateral_filter_h_from_hsv_differences',
        #  '{}_bilateral_filter_mean_color_diffs',
        #  '{}_bilateral_filter_n_color_diff',
        #  '{}_color_counter_norm_color_count',
        #  '{}_edges_detector_grayscale_edges_factor',
        #  '{}_hsv_analyser_hsv_var',
        #  '{}_hsv_analyser_saturation_distribution',
        #  '{}_hsv_analyser_sat_value_distribution',
        #  '{}_kmeans_segementator_hsv_differences_3',
        #  '{}_kmeans_segementator_hsv_differences_6',
        #  '{}_kmeans_segementator_hsv_differences_9',
        #  '{}_kmeans_segementator_hsv_differences_12',
        #  '{}_kmeans_segementator_mean_color_diffs_3',
        #  '{}_kmeans_segementator_mean_color_diffs_6',
        #  '{}_kmeans_segementator_mean_color_diffs_9',
        #  '{}_kmeans_segementator_mean_color_diffs_12'],
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
        #  '{}_gabor_filter']
        ['{}_bilateral_filter_h_from_hsv_differences',
         '{}_bilateral_filter_mean_color_diffs',
         '{}_bilateral_filter_n_color_diff',
         '{}_color_counter_norm_color_count',
         '{}_edges_detector_grayscale_edges_factor',
         '{}_kmeans_segementator_hsv_differences_15',
         '{}_kmeans_segementator_hsv_differences_25',
         '{}_kmeans_segementator_hsv_differences_35',
         '{}_kmeans_segementator_hsv_differences_45',
         '{}_kmeans_segementator_hsv_differences_55',
         '{}_gabor_filter']
    ]

    features_names = [
        # ['bilateral_filter'],
        # ['color_counter_edges_detector'],
        # ['hsv_analyser_hsv_var'],
        # ['hsv_analyser_saturation_distribution'],
        # ['hsv_analyser_sat_value_distribution'],
        # ['scalar'],
        # ['hsv_analyser'],
        # ['scalar_hsv'],
        # ['scalar_hsv_kmeans'],
        # ['scalar_hsv_kmeans3'],
        # ['gabor_filter'],
        # ['scalar_gabor'],
        # ['scalar_hsv_gabor'],
        # ['scalar_hsv_kmeans_gabor'],
        ['scalar_kmeans_gabor']
    ]

    dataset_sizes = [150, 300, 450, 608]  # combined
    # dataset_sizes = [120, 240, 363]             # memes

    y_labels = ["cartoon", "painting", "photo", "text"]
    im_types = ["pics", "memes"]
    # im_types = ["memes"]
    stands = [True, False]
    im_type = "combined"

    create_combined_features(filenames_list, features_names, im_types)
    experiments_on_dataset_size(features_names, dataset_sizes, stands, im_type, y_labels)
