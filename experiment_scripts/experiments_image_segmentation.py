import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader.utils import load_image_by_cv2
from image_segmentation.hough_lines import HoughLines
from settings import DATA_PATH


def hough_tests_for_all_params(plot_stats=True, plot=False, verbose=True):
    images_path = os.path.join(DATA_PATH, "base_dataset", "segmentation", "tests")
    file_names_full = glob.glob(images_path + '/*')
    file_names = [fn.split('/')[-1] for fn in file_names_full]
    labels = [int(fn.split('.')[-2][-1]) for fn in file_names]

    if plot_stats:
        plot_segments_distribution(labels)

    cv2_images = []
    for file_name in file_names_full:
        image = load_image_by_cv2(file_name)
        cv2_images.append(image)

    min_canny_thresholds = [50]  # [40, 50, 60]
    max_canny_thresholds = [180, 185, 190, 195, 200]  # [125, 150, 175]
    min_mask_thresholds = [235]  # [235, 240, 245]
    max_mask_thresholds = [260]  # [250, 260, 270]

    df = pd.DataFrame(columns=['time', 'min_canny', 'max_canny', 'min_mask', 'max_mask',
                               'auto_fun', 'auto_param1', 'auto_param2', 'num_wrong', 'num_all'])

    timestamp = time.time()

    for min_ct in min_canny_thresholds:
        for max_ct in max_canny_thresholds:
            for min_mask in min_mask_thresholds:
                for max_mask in max_mask_thresholds:
                    hl = HoughLines(min_ct, max_ct, min_mask, max_mask)
                    cv2_edges = get_edges_for_given_params(cv2_images, hl)

                    funs = [min]
                    params1 = [7, 7.5, 8]  # [8, 8.5, 9, 9.5, 10, 10.5, 11]
                    params2 = [0.9]  # [0.8, 0.85, 0.9]

                    for fun in funs:
                        fun_name = fun.__name__
                        for param2 in params2:
                            for param1 in params1:
                                if verbose:
                                    print('------------------------------------')
                                    print(f'Params: canny_thresholds = ({hl.min_canny_threshold}, {hl.max_canny_threshold}), '
                                          f'mask_thresholds = ({hl.min_mask_threshold}, {hl.max_mask_threshold}), '
                                          f'function = {fun_name} with MIN_LINE_LEN = {param1} and param2 = {param2}')

                                autotunes = count_autotune_param_for_all_edges(cv2_edges, fun=fun, param1=param1, param2=param2)
                                n_wrong, n_all, t = count_wrongly_labeled_images(hl, cv2_images, cv2_edges, autotunes, labels, file_names, verbose=verbose)

                                df = df.append({'time': t, 'min_canny': min_ct, 'max_canny': max_ct, 'min_mask': min_mask,
                                                'max_mask': max_mask, 'auto_fun': fun_name, 'auto_param1': param1,
                                                'auto_param2': param2, 'num_wrong': n_wrong, 'num_all': n_all}, ignore_index=True)

                    save_df_to_csv(df, timestamp, min_ct, max_ct, min_mask, max_mask)

    return df, timestamp


def save_df_to_csv(df, timestamp, min_ct=None, max_ct=None, min_mask=None, max_mask=None):
    path = os.path.join(DATA_PATH, 'results', 'segmentation')
    fn_prefix = f'experiments_{str(timestamp).split(".")[0]}'
    file_name = fn_prefix + '.csv' if min_ct is None else fn_prefix + f'_{min_ct}_{max_ct}_{min_mask}_{max_mask}.csv'
    df.to_csv(os.path.join(path, file_name), index=False)


def count_autotune_param_for_all_edges(cv2_edges, fun=min, param1=10, param2=0.8):
    autotunes = []
    for i in range(len(cv2_edges)):
        x, y = cv2_edges[i].shape
        norm_edges = int(np.sum(cv2_edges[i]) / 255) / (x * y)
        auto_min_line_len = min(int(norm_edges * fun(x, y) * param1), int(param2 * fun(x, y)))
        autotunes.append(auto_min_line_len)
    return autotunes


def get_edges_for_given_params(cv2_images, hl):
    cv2_edges = []
    for i in range(len(cv2_images)):
        image = cv2_images[i].copy()
        edges = hl.get_edges(image)
        cv2_edges.append(edges)
    return cv2_edges


def count_wrongly_labeled_images(hl, cv2_images, cv2_edges, autotunes, labels, file_names, plot=False, verbose=True):
    time_start = time.time()
    wrong_label = 0

    for i, (label, file_name) in enumerate(zip(labels, file_names)):
        image = cv2_images[i].copy()
        edges = cv2_edges[i].copy()
        autotuned_min_lines = autotunes[i]
        im, norm_edges, auto_min_line_len = hl.get_image_with_lines(image, edges, min_line_len=autotuned_min_lines)
        boxes = hl.get_bounding_boxes(im, plot=False)
        if len(boxes) != label:
            if plot:
                hl.get_bounding_boxes(im, plot=plot, plot_title=f'{len(boxes)} instead {label} bounding boxes')
            wrong_label += 1
    time_stop = time.time()

    if verbose:
        print(f'Wrongly labeled images: {wrong_label}/{len(file_names)}\n')
        # print(f'Time: {round(time_stop - time_start, 2)} sec')

    return wrong_label, len(file_names), round(time_stop - time_start, 2)


def plot_segments_distribution(labels):
    labels_no, labels_count = np.unique(labels, return_counts=True)
    labels_no = ['1', '2', '3', '4', '5 and more']
    labels_count = list(labels_count[:4]) + [sum(labels_count[4:])]
    plt.bar(labels_no, labels_count)
    plt.xlabel('number of pictures on single meme')
    plt.ylabel('number of memes')
    plt.show()


if __name__ == "__main__":
    df, timestamp = hough_tests_for_all_params(plot_stats=True, plot=False)
    save_df_to_csv(df, timestamp)
