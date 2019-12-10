import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import MiniBatchKMeans


class KMeansSegmentator():
    """
    Class counts euclidean distance of colors before and after some filters/segmentation applied to original picture.
    """
    def __init__(self, n_colors):
        """
        :param n_colors: number of colors for segmentation
        """
        self.num_colors = n_colors

    def apply_kmeans(self, image):
        """
        Applies bilateral filtering on the image
        :param image: image as numpy array
        :return: image as numpy array
        """
        (h, w) = image.shape[:2]
        img = image.reshape(-1, 3)
        clt = MiniBatchKMeans(n_clusters=self.num_colors)
        labels = clt.fit_predict(img)
        kmeans_image = clt.cluster_centers_.astype("uint8")[labels]
        return kmeans_image.reshape((h, w, 3))

    def mean_color_diffs(self, image, kmeans_image=None):
        """
        # Same method as in BilateralFilter
        Counts mean color differences within all colors.
        :param image: original image as matrix cv2_image
        :param kmeans_image: image as matrix after bilateral filter
        :return: mean difference (any number)
        """
        if kmeans_image is not None:
            im_km = kmeans_image.reshape((-1, 3))
        else:
            im_km = self.apply_kmeans(image).reshape((-1, 3))
        im = image.reshape((-1, 3))

        diff = np.mean(im - im_km, axis=1)
        return np.mean(diff)

    def _transform_to_hsv(self, image):
        """
        Transforms image from RGB to HSV.
        :param image: numpy array representing image in RGB
        :return: same array but in HSV
        """
        return mpl.colors.rgb_to_hsv(image)

    def hsv_differences(self, image, kmeans_image):
        """
        Counts differences separately between all HSV channels before and after kMeans.
        :param image: numpy array in RGB
        :param kmeans_image: numpy array of transformed image in RGB
        :return: differences on every channel: H, S and V
                 H channel should *probably* be best for image classification
        """
        image_hsv = self._transform_to_hsv(image)
        quant_hsv = self._transform_to_hsv(kmeans_image)

        h_diff = np.mean(image_hsv[:, :, 0] - quant_hsv[:, :, 0])
        s_diff = np.mean(image_hsv[:, :, 1] - quant_hsv[:, :, 1])
        v_diff = np.mean(image_hsv[:, :, 2] - quant_hsv[:, :, 2])
        return h_diff, s_diff, v_diff
