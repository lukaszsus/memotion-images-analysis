import numpy as np
import os
import matplotlib.pyplot as plt
from unittest import TestCase
from data_loader.utils import load_image_as_array
from feature_extraction.hsv_analyser import HsvAnalyser
from settings import DATA_PATH


class TestHsvAnalyser(TestCase):
    def test_angle_cos_var(self):
        file_path = os.path.join(DATA_PATH, "photo")
        file_path = os.path.join(file_path, "funny-game-of-thrones-memes-fb__700.jpg")
        im = load_image_as_array(file_path)
        print(im.shape)
        spatial_variance = HsvAnalyser()
        features = spatial_variance.hsv_var(im, (5, 5))
        expected = np.array([8.20562953e-04, 4.93038066e-32, 2.67054637e-09, 1.96633129e-07,
                             9.88124716e-07, 6.82860813e-05, 1.10551960e-02, 0.00000000e+00,
                             1.71209412e-05, 1.81389472e-04, 1.07205008e-03, 3.37056916e-02,
                             2.28070797e-02, 2.46059208e-06, 8.48904268e-05, 5.61654748e-04,
                             5.68519800e-03, 1.17176111e-01])
        print(np.mean(features - expected))
        self.assertEqual(str(features), str(expected))

    def test_saturation_distribution(self):
        file_photo = os.path.join(DATA_PATH, "photo")
        file_photo = os.path.join(file_photo, "funny-game-of-thrones-memes-fb__700.jpg")
        im_photo = load_image_as_array(file_photo)
        print(im_photo.shape)

        file_paint = os.path.join(DATA_PATH, "painting")
        file_paint = os.path.join(file_paint, "5d646e19b30e1.jpeg")
        im_paint = load_image_as_array(file_paint)
        print(im_paint.shape)

        spatial_variance = HsvAnalyser()
        features_photo = spatial_variance.saturation_distribution(im_photo)
        features_paint = spatial_variance.saturation_distribution(im_paint)

        print(features_paint.shape)

        x = np.arange(len(features_photo))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.bar(x - width / 2, features_photo, width, label='photo')
        ax.bar(x + width / 2, features_paint, width, label='painting')
        fig.legend()
        plt.show()
        # Histograms have a very high not saturated pixels.
        # It is probably cased by white and black background and text in memes.

        self.assertTrue(True)

    def test_sat_value_distribution(self):
        file_photo = os.path.join(DATA_PATH, "photo")
        file_photo = os.path.join(file_photo, "funny-game-of-thrones-memes-fb__700.jpg")
        im_photo = load_image_as_array(file_photo)
        print(im_photo.shape)

        file_paint = os.path.join(DATA_PATH, "painting")
        file_paint = os.path.join(file_paint, "5d646e19b30e1.jpeg")
        im_paint = load_image_as_array(file_paint)
        print(im_paint.shape)

        spatial_variance = HsvAnalyser()
        features_photo = spatial_variance.sat_value_distribution(im_photo)
        features_paint = spatial_variance.sat_value_distribution(im_paint)

        x = np.arange(len(features_photo))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.bar(x - width / 2, features_photo, width, label='photo')
        ax.bar(x + width / 2, features_paint, width, label='painting')
        fig.legend()
        plt.show()
        # Histograms have a very high not saturated pixels.
        # It is probably cased by white and black background and text in memes.
        self.assertTrue(True)
