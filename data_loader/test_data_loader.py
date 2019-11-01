import os
import numpy as np
from unittest import TestCase
from data_loader.utils import load_image_as_norm_array, load_image_as_array
from settings import DATA_PATH


class TestDataLoader(TestCase):
    def test_load_single_file_as_norm_array(self):
        path_to_dir = os.path.join(DATA_PATH, "photo")
        files = os.listdir(path_to_dir)
        file_path = os.path.join(path_to_dir, files[0])
        im = load_image_as_norm_array(file_path)
        self.assertEqual(type(im), np.ndarray)

    def test_load_image_as_array(self):
        path_to_dir = os.path.join(DATA_PATH, "photo")
        files = os.listdir(path_to_dir)
        file_path = os.path.join(path_to_dir, files[0])
        im = load_image_as_array(file_path)
        self.assertEqual(type(im), np.ndarray)