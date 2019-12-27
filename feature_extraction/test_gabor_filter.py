import os
import numpy as np
from unittest import TestCase

from data_loader.utils import load_image_by_cv2
from feature_extraction.gabor_filter import GaborFilter
from settings import DATA_PATH


class TestGaborFilter(TestCase):
    def test_apply_filter(self):
        file_photo = os.path.join(DATA_PATH, "base_dataset/photo/pics/pexels-photo-2873992.jpeg")
        print(file_photo)
        im_photo = load_image_by_cv2(file_photo)

        file_cartoon = os.path.join(DATA_PATH, "base_dataset/cartoon/pics/cartoon_1.jpeg")
        print(file_cartoon)
        im_cartoon = load_image_by_cv2(file_cartoon)

        gabor_filter = GaborFilter()
        photo_feats = gabor_filter.apply_filter(im_photo)
        cartoon_feats = gabor_filter.apply_filter(im_cartoon)

        print(photo_feats)
        print(cartoon_feats)
        print(np.linalg.norm(photo_feats))
        print(np.linalg.norm(cartoon_feats))
        self.assertGreater(np.linalg.norm(photo_feats), np.linalg.norm(cartoon_feats))
