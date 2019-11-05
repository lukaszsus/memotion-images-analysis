import os
import numpy as np
from unittest import TestCase

from data_loader.utils import load_image_as_array
from feature_extraction.colors_counter import ColorCounter
from settings import DATA_PATH


class TestColorCounter(TestCase):
    def test_norm_color_count(self):
        file_photo = os.path.join(DATA_PATH, "photo")
        file_photo = os.path.join(file_photo, "funny-game-of-thrones-memes-fb__700.jpg")
        im_photo = load_image_as_array(file_photo)
        # print(im_photo.shape)

        file_paint = os.path.join(DATA_PATH, "painting")
        file_paint = os.path.join(file_paint, "5d646e19b30e1.jpeg")
        im_paint = load_image_as_array(file_paint)
        # print(im_paint.shape)

        file_cartoon = os.path.join(DATA_PATH, "cartoon")
        file_cartoon = os.path.join(file_cartoon, "cartoon_1.jpg")
        im_cartoon = load_image_as_array(file_cartoon)

        color_counter = ColorCounter()
        features_photo = color_counter.norm_color_count(im_photo)
        features_paint = color_counter.norm_color_count(im_paint)
        features_cartoon = color_counter.norm_color_count(im_cartoon)

        diff_painting = features_photo - features_paint
        diff_cartoon = features_photo - features_cartoon

        print(diff_painting)
        self.assertGreater(0, diff_painting)

        print(diff_cartoon)
        self.assertGreater(diff_cartoon, 0)
