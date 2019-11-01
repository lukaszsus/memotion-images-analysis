import numpy as np
import os
from unittest import TestCase

from data_loader.utils import load_image_as_array
from feature_extraction.spatial_variance import SpatialVariance
from settings import DATA_PATH


class TestSpatialVariance(TestCase):
    def test_angle_cos_var(self):
        path_to_dir = os.path.join(DATA_PATH, "photo")
        files = os.listdir(path_to_dir)
        file_path = os.path.join(path_to_dir, files[0])
        im = load_image_as_array(file_path)
        print(im.shape)
        spatial_variance = SpatialVariance()
        features = spatial_variance.angle_cos_var(im, (5, 5))
        print(features)
        self.assertGreater(np.sum(features), 0)



