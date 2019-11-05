import os
import numpy as np
from unittest import TestCase

from data_loader.utils import load_image_by_cv2
from feature_extraction.bilateral_filter import BilateralFilter
from settings import DATA_PATH


class TestBilateralFilter(TestCase):
    def test_mean_color_diffs(self):
        file_photo = os.path.join(DATA_PATH, "photo/funny-game-of-thrones-memes-fb__700.jpg")
        im_photo = load_image_by_cv2(file_photo)

        file_cartoon = os.path.join(DATA_PATH, "cartoon/cartoon_1.jpg")
        im_cartoon = load_image_by_cv2(file_cartoon)

        bilateral_filter = BilateralFilter(30, 50, 50)
        color_diffs_photo = bilateral_filter.mean_color_diffs(im_photo)
        color_diffs_cartoon = bilateral_filter.mean_color_diffs(im_cartoon)

        self.assertGreater(color_diffs_photo, color_diffs_cartoon)

