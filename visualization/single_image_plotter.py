import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from settings import DATA_PATH


class SingleImagePlotter():
    """
    Shows plots connected with given image.
    """
    def __init__(self):
        pass

    def plot_image_from_path(self, directory, img_name):
        """
        Loads image from given path and shows it.
        :param directory: directory containing specific image type eg. photo, cartoon
        :param img_name: image name including extension
        """
        path = os.path.join(DATA_PATH, directory)
        img_path = os.path.join(path, img_name)

        img = cv2.imread(img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        plt.figure(figsize=(int(w/100), int(h/100)))
        plt.axis("off")
        plt.imshow(image)
        plt.show()

    def plot_image(self, image):
        """
        Simply plots image.
        :param image: image as numpy array
        """
        try:
            h, w, _ = image.shape
        except Exception:
            h, w = image.shape
        # plt.figure(figsize=(int(w/100), int(h/100)))
        # plt.figure(figsize=(20, 10))
        plt.axis("off")
        plt.imshow(image)
        plt.show()

    def plot_compared_images(self, original_image, transformed_image):
        """
        Simply plots image.
        :param image: image as numpy array
        """
        h, w, _ = original_image.shape
        plt.figure(figsize=(int(w/100) * 2, int(h/100)))
        plt.axis("off")
        white_space = np.full(shape=(h, int(w/100), 3), fill_value=255)
        plt.imshow(np.hstack([original_image, white_space, transformed_image]))
        plt.show()

    def plot_image_hist(self, image):
        """
        Shows histogram of image pixels.
        :param image: image as numpy array
        """
        histr = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(histr)
        plt.xlim([0, 256])
        plt.show()

    def plot_image_hist_rgb(self, image):
        """
        Shows coloured histogram of image pixels for each RGB chanel.
        :param image: image as numpy array
        """
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()

