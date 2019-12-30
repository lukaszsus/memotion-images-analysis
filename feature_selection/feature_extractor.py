import os
import pickle

import numpy as np
import pandas as pd
from data_loader.utils import load_image_by_cv2
from feature_extraction.bilateral_filter import BilateralFilter
from feature_extraction.color_counter import ColorCounter
from feature_extraction.edges_detector import EdgesDetector
from feature_extraction.gabor_filter import GaborFilter
from feature_extraction.hsv_analyser import HsvAnalyser
from feature_extraction.kmeans_segmentator import KMeansSegmentator
from functools import partial
from tqdm import tqdm


class FeatureExtractor():
    """
    Class creates pickle file with features and labels for every feature extractors' method.
    """
    def __init__(self):
        self.bilateral_filter: BilateralFilter = None
        self.color_counter: ColorCounter = None
        self.edges_detector: EdgesDetector = None
        self.hsv_analyser: HsvAnalyser = None
        self.kmeans_segementator = None
        self.gabor_filter = None

        self.extraction_pipelines: list = None
        self.feature_names: list = None

        self._initialize()

    def create_features_from_dir(self, src_path: str, im_type: str, save_path: str=None):
        """
        Creates pickles with features. Basically, one file contains output of one feature extraction method.
        It assumes that src_path has separated subdirectories for every class, for example:
        src_path:
        |-----cartoon
        |-------------pics
        |-------------memes
        |-----other
        |-------------pics
        |-------------memes
        |-----painting
        |-------------pics
        |-------------memes
        |-----photo
        |-------------pics
        |-------------memes
        |-----text
        |-------------pics
        |-------------memes
        :param src_path: path to directory with images in subdirectories for every class
        :param im_type: 'memes' or 'pics'
        :param save_path: directory to save binaries in
        :return:
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i in range(len(self.extraction_pipelines)):
            self.extraction_pipeline = self.extraction_pipelines[i]()
            self.feature_name = self.feature_names[i]
            print("Feature name: {}".format(self.feature_name))
            self._create_feature_from_dir(src_path, im_type, save_path)

    def create_labels_from_dir(self, src_path: str, im_type: str, save_path: str=None):
        """
        Creates pickle with labels.
        It assumes that src_path has separated subdirectories for every class, for example:
        src_path:
        |-----cartoon
        |-------------pics
        |-------------memes
        |-----other
        |-------------pics
        |-------------memes
        |-----painting
        |-------------pics
        |-------------memes
        |-----photo
        |-------------pics
        |-------------memes
        |-----text
        |-------------pics
        |-------------memes
        :param src_path: path to directory with images in subdirectories for every class
        :param im_type: 'memes' or 'pics'
        :param save_path: directory to save binary in
        :return:
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.labels = os.listdir(src_path)
        self.labels.sort()
        targets = list()
        for i in range(len(self.labels)):
            label = self.labels[i]
            print("Label: {}".format(label))
            label_path = os.path.join(src_path, label)
            if im_type is not None:
                label_path = os.path.join(label_path, im_type)
            files = os.listdir(label_path)
            files.sort()
            for file_name in tqdm(files):
                targets.append(i)
        targets = np.array(targets)
        if len(targets.shape) == 1:
            features = np.reshape(targets, (-1, 1))

        if save_path is not None:
            self._save_binary(targets, os.path.join(save_path, im_type + "_labels"))

    def create_path_list_from_dir(self, src_path: str, im_type: str, save_path: str=None):
        """
        Creates a list with files path.
        It assumes that src_path has separated subdirectories for every class, for example:
        src_path:
        |-----cartoon
        |-------------pics
        |-------------memes
        |-----other
        |-------------pics
        |-------------memes
        |-----painting
        |-------------pics
        |-------------memes
        |-----photo
        |-------------pics
        |-------------memes
        |-----text
        |-------------pics
        |-------------memes
        :param src_path: path to directory with images in subdirectories for every class
        :param im_type: 'memes' or 'pics'
        :param save_path: directory to save binary in
        :return:
        """
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.labels = os.listdir(src_path)
        self.labels.sort()
        file_paths = list()
        for i in range(len(self.labels)):
            label = self.labels[i]
            print("Label: {}".format(label))
            label_path = os.path.join(src_path, label)
            if im_type is not None:
                label_path = os.path.join(label_path, im_type)
            files = os.listdir(label_path)
            files.sort()
            for file_name in tqdm(files):
                file_path = os.path.join(label_path, file_name)
                file_path = file_path.replace(src_path, "")
                file_paths.append(file_path)

        if save_path is not None:
            self._save_binary(file_paths, os.path.join(save_path, im_type + "_paths"))

    def _create_feature_from_dir(self, src_path: str, im_type: str, save_path: str=None):
        """
        Creates pickle file with feature for every example in directory.
        Parameters the same as in create_features_from_dir.
        """
        self.labels = os.listdir(src_path)
        self.labels.sort()
        features = list()
        for i in range(len(self.labels)):
            label = self.labels[i]
            print("Label: {}".format(label))
            label_path = os.path.join(src_path, label)
            if im_type is not None:
                label_path = os.path.join(label_path, im_type)
            files = os.listdir(label_path)
            files.sort()
            for file_name in tqdm(files):
                file_path = os.path.join(label_path, file_name)
                im = load_image_by_cv2(file_path)
                features.append(self.extraction_pipeline(im))
        features = np.array(features)
        if len(features.shape) == 1:
            features = np.reshape(features, (-1, 1))

        if save_path is not None:
            self._save_binary(features, os.path.join(save_path, im_type + "_" + self.feature_name))

    def _save_binary(self, features, file_path):
        base, ext = os.path.splitext(file_path)
        if ext != ".pickle" or ext != ".pkl":
            file_path += ".pickle"
        with open(file_path, 'wb') as file:
            pickle.dump(features, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _initialize(self):
        self.bilateral_filter = BilateralFilter(30, 50, 50)
        self.color_counter = ColorCounter()
        self.edges_detector = EdgesDetector()
        self.hsv_analyser = HsvAnalyser()
        self.kmeans_segementator = None
        self.gabor_filter = GaborFilter()

        _kmeans_mean_color_diffs_3 = partial(self._kmeans_segmentator_mean_color_diffs, n_colors=3)
        _kmeans_mean_color_diffs_6 = partial(self._kmeans_segmentator_mean_color_diffs, n_colors=6)
        _kmeans_mean_color_diffs_9 = partial(self._kmeans_segmentator_mean_color_diffs, n_colors=9)
        _kmeans_mean_color_diffs_12 = partial(self._kmeans_segmentator_mean_color_diffs, n_colors=12)
        _kmeans_mean_color_diffs_15 = partial(self._kmeans_segmentator_mean_color_diffs, n_colors=15)
        _kmeans_mean_color_diffs_25 = partial(self._kmeans_segmentator_mean_color_diffs, n_colors=25)
        _kmeans_mean_color_diffs_35 = partial(self._kmeans_segmentator_mean_color_diffs, n_colors=35)
        _kmeans_mean_color_diffs_45 = partial(self._kmeans_segmentator_mean_color_diffs, n_colors=45)
        _kmeans_mean_color_diffs_55 = partial(self._kmeans_segmentator_mean_color_diffs, n_colors=55)

        _kmeans_mean_hsv_diffs_3 = partial(self._kmeans_segmentator_hsv_differences, n_colors=3)
        _kmeans_mean_hsv_diffs_6 = partial(self._kmeans_segmentator_hsv_differences, n_colors=6)
        _kmeans_mean_hsv_diffs_9 = partial(self._kmeans_segmentator_hsv_differences, n_colors=9)
        _kmeans_mean_hsv_diffs_12 = partial(self._kmeans_segmentator_hsv_differences, n_colors=12)
        _kmeans_mean_hsv_diffs_15 = partial(self._kmeans_segmentator_hsv_differences, n_colors=15)
        _kmeans_mean_hsv_diffs_25 = partial(self._kmeans_segmentator_hsv_differences, n_colors=25)
        _kmeans_mean_hsv_diffs_35 = partial(self._kmeans_segmentator_hsv_differences, n_colors=35)
        _kmeans_mean_hsv_diffs_45 = partial(self._kmeans_segmentator_hsv_differences, n_colors=45)
        _kmeans_mean_hsv_diffs_55 = partial(self._kmeans_segmentator_hsv_differences, n_colors=55)

        self.extraction_pipelines = [
            # self._basic_method_wrapper(self.bilateral_filter.mean_color_diffs),
            # self._basic_method_wrapper(self.bilateral_filter.n_color_diff),
            # self._basic_method_wrapper(self.bilateral_filter.h_from_hsv_differences),
            # self._basic_method_wrapper(self.color_counter.norm_color_count),
            # self._basic_method_wrapper(self.edges_detector.grayscale_edges_factor),
            # self._basic_method_wrapper(self.hsv_analyser.hsv_var),
            # self._basic_method_wrapper(self.hsv_analyser.saturation_distribution),
            # self._basic_method_wrapper(self.hsv_analyser.sat_value_distribution),
            # _kmeans_mean_color_diffs_15,
            # _kmeans_mean_color_diffs_25,
            # _kmeans_mean_color_diffs_35,
            # _kmeans_mean_color_diffs_45,
            # _kmeans_mean_color_diffs_55,
            # _kmeans_mean_hsv_diffs_15,
            # _kmeans_mean_hsv_diffs_25,
            # _kmeans_mean_hsv_diffs_35,
            # _kmeans_mean_hsv_diffs_45,
            # _kmeans_mean_hsv_diffs_55,
            # _kmeans_mean_color_diffs_3,
            # _kmeans_mean_color_diffs_6,
            # _kmeans_mean_color_diffs_9,
            # _kmeans_mean_color_diffs_12,
            # _kmeans_mean_hsv_diffs_3,
            # _kmeans_mean_hsv_diffs_6,
            # _kmeans_mean_hsv_diffs_9,
            # _kmeans_mean_hsv_diffs_12,
            self._basic_method_wrapper(self.gabor_filter.apply_filter),
        ]

        self.feature_names = [
            # "bilateral_filter_mean_color_diffs",
            # "bilateral_filter_n_color_diff",
            # "bilateral_filter_h_from_hsv_differences",
            # "color_counter_norm_color_count",
            # "edges_detector_grayscale_edges_factor",
            # "hsv_analyser_hsv_var",
            # "hsv_analyser_saturation_distribution",
            # "hsv_analyser_sat_value_distribution",
            # "kmeans_segementator_mean_color_diffs_15",
            # "kmeans_segementator_mean_color_diffs_25",
            # "kmeans_segementator_mean_color_diffs_35",
            # "kmeans_segementator_mean_color_diffs_45",
            # "kmeans_segementator_mean_color_diffs_55",
            # "kmeans_segementator_hsv_differences_15",
            # "kmeans_segementator_hsv_differences_25",
            # "kmeans_segementator_hsv_differences_35",
            # "kmeans_segementator_hsv_differences_45",
            # "kmeans_segementator_hsv_differences_55",
            # "kmeans_segementator_mean_color_diffs_3",
            # "kmeans_segementator_mean_color_diffs_6",
            # "kmeans_segementator_mean_color_diffs_9",
            # "kmeans_segementator_mean_color_diffs_12",
            # "kmeans_segementator_hsv_differences_3",
            # "kmeans_segementator_hsv_differences_6",
            # "kmeans_segementator_hsv_differences_9",
            # "kmeans_segementator_hsv_differences_12",
            "gabor_filter"
        ]

        if len(self.extraction_pipelines) != len(self.feature_names):
            raise ValueError("Fields self.extraction_pipelines and self.feature_names must be the same length.")

    def _basic_method_wrapper(self, m: classmethod):
        def pipeline(im: np.array):
            return m(im)

        def wrapper():
            return pipeline
        return wrapper

    def _kmeans_segmentator_mean_color_diffs(self, n_colors):
        self.kmeans_segementator = KMeansSegmentator(n_colors=n_colors)

        def pipeline(im: np.array):
            return self.kmeans_segementator.mean_color_diffs(im)
        return pipeline

    def _kmeans_segmentator_hsv_differences(self, n_colors):
        self.kmeans_segementator = KMeansSegmentator(n_colors=n_colors)

        def pipeline(im: np.array):
            return self.kmeans_segementator.hsv_differences(im, self.kmeans_segementator.apply_kmeans(im))
        return pipeline

