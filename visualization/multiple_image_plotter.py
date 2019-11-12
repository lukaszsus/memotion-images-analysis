import os
import cv2
from matplotlib import pyplot as plt
import glob
import numpy as np
# import seaborn as sns

from user_settings import DATA_PATH


class MultipleImagePlotter():
    """
    Shows plots connected with given image-type eg. cartoon or photo.
    """
    def __init__(self):
        pass

    def plot_images(self, directory):
        path = os.path.join(DATA_PATH, directory)
        img_names = glob.glob(f"{path}/*")
        sqrt_sub = np.sqrt(len(img_names))
        sub_size = int(sqrt_sub) if sqrt_sub == int(sqrt_sub) else int(sqrt_sub)+1

        fig, ax = plt.subplots(sub_size, sub_size, figsize=(10, 10))
        for i, img_name in enumerate(img_names):
            img = cv2.imread(img_name)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            x, y = int(i/sub_size), i % sub_size

            ax[x, y].imshow(image)
            ax[x, y].grid(False)
            ax[x, y].axis('off')

        for j in range(len(img_names), sub_size*sub_size):
            ax[int(j/sub_size), j%sub_size].axis('off')

        plt.show()

    def plot_avg_histogram(self, directory, scale='linear'):
        path = os.path.join(DATA_PATH, directory)
        img_names = glob.glob(f"{path}/*")
        all_hists = np.zeros(shape=(256, 1))

        for i, img_name in enumerate(img_names):
            img = cv2.imread(img_name)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            histr = cv2.calcHist([image], [0], None, [256], [0, 256])
            all_hists += histr

        plt.plot(all_hists)
        plt.xlim([0, 256])
        plt.yscale(scale)
        plt.show()
