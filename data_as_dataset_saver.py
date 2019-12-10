"""
Saves all data from given directory as a dataset - pairs (x, y),
where x is a vector of given features and y is a label.
"""
import os
import glob
from time import time
import pickle as pkl

from data_loader.utils import load_image_by_cv2
from feature_extraction.bilateral_filter import BilateralFilter
from settings import PROJECT_PATH

from user_settings import DATA_PATH

LABELS = ('cartoon', 'painting', 'photo', 'text')


class DataSaver():
    def __init__(self, dataset_type):
        # TODO use Łukaszs code to do this, after merge :D
        self.data_type = dataset_type
        self.path = os.path.join(DATA_PATH, 'datasets_pkl')

    def _load_data_from_one_directory(self, label_name):
        path = os.path.join(DATA_PATH, label_name)
        data_path = os.path.join(path, self.data_type)
        image_names = [f for f in glob.glob(f'{data_path}/*') if os.path.isfile(f)]

        images_cv2 = []
        for img_name in image_names:
            images_cv2.append(load_image_by_cv2(img_name))

        return images_cv2

    def _get_dataset_from_one_directory(self, label_name, images_cv2):
        ''' FEATURES '''
        bf = BilateralFilter(30, 50, 50)
        x_data, y_data = [], [LABELS.index(label_name) for _ in images_cv2]
        for i, image in enumerate(images_cv2):
            x_data.append([])
            bil_image = bf.apply_filter(image)
            x_data[i].append(bf.h_from_hsv_differences(images_cv2[i], bil_image))
            x_data[i].append(bf.n_color_diff(images_cv2[i], bil_image))

        return x_data, y_data

    def get_dataset_from_all_directories(self):
        x, y = [], []
        for label_name in LABELS:
            print(f'{label_name.capitalize()} starting...')
            t = time()
            images_csv = self._load_data_from_one_directory(label_name)
            x_data, y_data = self._get_dataset_from_one_directory(label_name, images_csv)
            x += x_data
            y += y_data
            print(f'  finishing after {round(time() - t, 3)} seconds')

        x_feature_names = ['bilateral_h_hsv_diff', 'bilateral_norm_color_diff']
        return x, y, x_feature_names, LABELS

    def save_dataset_to_file(self, x, y, x_feature_names, y_feature_names, dataset_name):
        fname = f"{self.data_type}_{dataset_name}.pkl"
        with open(os.path.join(self.path, fname), "wb") as fout:
            pkl.dump([x, y, x_feature_names, y_feature_names], fout, protocol=-1)