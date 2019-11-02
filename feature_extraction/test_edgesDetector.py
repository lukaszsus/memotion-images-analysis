import os
import numpy as np
from unittest import TestCase

from data_loader.utils import load_image_as_array
from feature_extraction.edges_detector import EdgesDetector
from settings import DATA_PATH


class TestEdgesDetector(TestCase):
    def test_grayscale_edges_factor(self):
        file_photo = os.path.join(DATA_PATH, "photo")
        file_photo = os.path.join(file_photo, "funny-game-of-thrones-memes-fb__700.jpg")
        im_photo = load_image_as_array(file_photo)
        print(im_photo.shape)

        file_paint = os.path.join(DATA_PATH, "painting")
        file_paint = os.path.join(file_paint, "5d646e19b30e1.jpeg")
        im_paint = load_image_as_array(file_paint)
        print(im_paint.shape)

        edges_detector = EdgesDetector()
        features_photo = edges_detector.grayscale_edges_factor(im_photo)
        features_paint = edges_detector.grayscale_edges_factor(im_paint)

        diff = np.mean(features_photo - features_paint)

        print(diff)
        self.assertGreater(diff, 0)
