import cv2
import numpy as np
import matplotlib.pyplot as plt


class EdgesDetector():
    """
    Class counts proportion of edges' pixels after converting image to grayscale and edges' pixel in rgb model.
    """
    def __init__(self, thresholds = None):
        if thresholds is None:
            self.thresholds = [(25, 100),
                               (50, 100),
                               (50, 150),
                               (100, 150),
                               (100, 200),
                               (150, 200),
                               (150, 225)]
        else:
            self.thresholds = thresholds

    def grayscale_edges_factor(self, image):
        """
        Counts proportion of edges' pixels after converting image to grayscale and edges' pixel in rgb model.
        Uses Canny algorithm of edge detection. Canny has two parameters: lower and upper threshold.
        Method do Canny algorithm using several pairs of thresholds defined in self.thresholds.
        :param image: image to process
        :return: vector of proportions created using different thresholds pairs defined in self.thresholds
        """
        proportions = list()
        im_color = np.uint8(image)
        im_gray = np.uint8(np.dot(image[..., :3], [0.299, 0.587, 0.114]))
        for threshold in self.thresholds:
            edges_color = cv2.Canny(im_color, *threshold)      # arbitrary chosen thresholds
            edges_gray = cv2.Canny(im_gray, *threshold)

            n_pix_edges_color = np.sum(edges_color / 255)
            n_pix_edges_gray = np.sum(edges_gray / 255)

            proportions.append(n_pix_edges_gray / n_pix_edges_color)

        return np.array(proportions)
