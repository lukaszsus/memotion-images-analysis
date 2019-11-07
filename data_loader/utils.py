import os

import cv2
import numpy as np
from PIL import Image
from settings import DATA_PATH
import matplotlib.pyplot as plt


def load_image_as_norm_array(file_path):
    """
    Funtion load single image from file.
    :param file_path:
    :return: image as numpy array
    """
    path = os.path.join(DATA_PATH, file_path)
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype=np.uint8)
    data = data / 255.0   # normalization
    return data


def load_image_as_array(file_path):
    """
    Funtion load single image from file.
    :param file_path:
    :return: image as numpy array
    """
    path = os.path.join(DATA_PATH, file_path)
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype=np.uint8)
    return data


def load_image_by_cv2(file_path):
    """
    Function load single image from file using cv2 method. Probably still loads image in BGR...
    :param file_path:
    :return: image as numpy array
    """
    path = os.path.join(DATA_PATH, file_path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
