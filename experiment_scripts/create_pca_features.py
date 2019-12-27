import numpy as np
from sklearn.decomposition import PCA


from experiment_scripts.experiments_for_multiple_classifiers import load_y_from_file, format_filenames, \
    load_x_y_from_files
from experiment_scripts.experiments_on_dataset_size import save_x_to_file, save_y_to_file


def create_pca_features(filenames_list, out_n_features):
    pass


if __name__ == '__main__':
    filenames_list = [
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
         '{}_kmeans_segementator_mean_color_diffs_55',
         '{}_gabor_filter']
    ]

    out_feature_name = "pca_top_{}"
    im_types = ["pics", "memes"]
    pca_top_vars = [5, 10, 15, 25, 50]

    for im_type in im_types:
        dirname = im_type + "_feature_binaries"
        for filenames in filenames_list:
            for pca_top_var in pca_top_vars:
                filenames_formatted = format_filenames(filenames, im_type)
                x, x_labels = load_x_y_from_files(dirname, filenames_formatted)

                pca = PCA(n_components=pca_top_var)
                x = pca.fit_transform(x)
                print(f'Num of dims: {pca_top_var}')
                print(f'Explained variance ratio: {np.sum(pca.explained_variance_ratio_)}')

                save_x_to_file(dirname, im_type, out_feature_name.format(pca_top_var), x)
