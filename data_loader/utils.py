import os
import numpy as np
from PIL import Image
from settings import DATA_PATH


def load_image_as_norm_array(file_path):
    """
    Funtion load single image from file.
    :param file_path:
    :return: image as numpy array
    """
    path = os.path.join(DATA_PATH, file_path)
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype="int32")
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
    data = np.asarray(img, dtype="int32")
    return data
