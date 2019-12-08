import os
import pickle

import numpy as np
import pandas as pd
from data_loader.utils import load_image_by_cv2
from deprecated import deprecated
from feature_extraction.bilateral_filter import BilateralFilter
from feature_extraction.color_counter import ColorCounter
from feature_extraction.edges_detector import EdgesDetector
from feature_extraction.feature_namer import FeatureNamer
from feature_extraction.hsv_analyser import HsvAnalyser


@deprecated(reason="More efficient way implemented in FeatureExtractor class.")
class DatasetCreator():
    """
    Class creates pandas DataFrame with columns names the same as features names.
    I allows to create different combinations of features. You need to pass a feature creation pipeline.
    """
    def __init__(self):
        self.feature_namer = FeatureNamer()

        self.bilateral_filter: BilateralFilter = None
        self.color_counter: ColorCounter = None
        self.edges_detector: EdgesDetector = None
        self.hsv_analyser: HsvAnalyser = None

        self.extraction_pipeline: classmethod = None
        self.dataset: pd.DataFrame = None
        self.labels: list = None
        self.__rows: list = None

    def set_to_scalar_features_extractor(self):
        """
        Method which changes object state. It could be a static method in factory framework.
        It initializes object and set self.extraction_pipeline method to get same scalar features.
        Features are selected to create readable scatter pairplot.
        :return:
        """
        self.hsv_analyser = HsvAnalyser()

        extractors = [self.hsv_analyser.hsv_var]
        names = self.feature_namer.get_features_names(extractors)
        columns = ["label_index"] + names
        self.dataset = pd.DataFrame(columns=columns)
        self.__rows = list()

        def pipeline(image: np.array, label_index: int):
            features = [self.flatten(np.array(label_index))]
            self.hsv_analyser.set_image_and_convert_to_hsv(image)
            for extractor in extractors:
                features.append(self.flatten(extractor(image)))
            features = np.concatenate(features, axis=0)
            self.__rows.append(pd.DataFrame(columns=columns, data=self.make_row(features)))

        self.extraction_pipeline = pipeline

    def set_to_local_var_features_extractor(self):
        """
        Method which changes object state. It could be a static method in factory framework.
        It initializes object and set self.extraction_pipeline method to get local variance.
        :return:
        """
        self.bilateral_filter = BilateralFilter(30, 50, 50)
        self.color_counter = ColorCounter()
        self.edges_detector = EdgesDetector()

        extractors = [self.bilateral_filter.mean_color_diffs,
                      self.color_counter.norm_color_count,
                      self.edges_detector.grayscale_edges_factor]
        names = self.feature_namer.get_features_names(extractors)
        columns = ["label_index"] + names
        self.dataset = pd.DataFrame(columns=columns)
        self.__rows = list()

        def pipeline(image: np.array, label_index: int):
            features = [self.flatten(np.array(label_index))]
            for extractor in extractors:
                features.append(self.flatten(extractor(image)))
            features = np.concatenate(features, axis=0)
            self.__rows.append(pd.DataFrame(columns=columns, data=self.make_row(features)))

        self.extraction_pipeline = pipeline

    @deprecated(reason="More efficient way implemented in method create features from dir.")
    def create_dataset_from_dir(self, src_path: str):
        """
        Creates dataset from source path (directory) and saves it to dst_path (csv file).
        It assumes that src_path has separated subdirectories for every class, for example:
        src_path:
        |-----cartoon
        |-----other
        |-----painting
        |-----photo
        |-----text
        :param src_path: path to directory with images in subdirectories for every class
        :param dst_path: path to csv file
        :param feature_extractors: list of methods which takes only one argument: image as numpy array
        :return:
        """
        self.labels = os.listdir(src_path)
        for i in range(len(self.labels)):
            label = self.labels[i]
            label_path = os.path.join(src_path, label)
            files = os.listdir(label_path)
            for file_name in files:
                file_path = os.path.join(label_path, file_name)
                print(file_path)
                im = load_image_by_cv2(file_path)
                self.extraction_pipeline(im, i)

        self.dataset = pd.concat([self.dataset, pd.concat(self.__rows, ignore_index=True)], ignore_index=True)

    def flatten(self, value):
        """
        Methods resizes scalar or zero-dimentional numpay arrays to array size (1, 1).
        :return:
        """
        if not hasattr(value, "__len__"):
            value = np.array(value)
        value = value.reshape((-1, 1))
        return value

    def make_row(self, vector: np.array):
        """
        Transposes column vector to row.
        :param vector: numpy array - column vector
        :return:
        """
        return np.transpose(vector)

    def save(self, file_path):
        """
        Saves self.dataset to CSV file.
        :param file_path: path to CSV file
        :return:
        """
        if self.dataset is not None:
            self.dataset.to_csv(file_path, index=False)
        base, ext = os.path.splitext(file_path)
        with open(base + "_labels.pickle", 'wb') as file:
            pickle.dump(self.labels, file)
