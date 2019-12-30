import os
import numpy as np
from unittest import TestCase

from data_loader.utils import load_image_by_cv2
from image_segmentation.hough_lines import HoughLines
from settings import DATA_PATH


class TestHoughLines(TestCase):
    def test_no_of_bounding_boxes(self):
        file_name = 'stats.jpg'
        file_path = os.path.join("base_dataset", "segmentation", file_name)
        image = load_image_by_cv2(file_path)

        hl = HoughLines()
        im, norm_edges, auto_min_line_len = hl.get_image_with_lines(image)
        boxes = hl.get_bounding_boxes(im, plot=False)

        self.assertEqual(len(boxes), 3)
