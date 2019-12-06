import numpy as np
from time import time
import matplotlib as mpl
import cv2


class BilateralFilter():
    def __init__(self, neighbourhood, sigma_color, sigma_space):
        """
        :param neighbourhood: diameter of each pixel neighborhood that is used during filtering
        :param sigma_color: larger value means that farther colors within the pixel neighborhood will be other together,
               resulting in larger areas of semi-equal color
        :param sigma_space: larger value means that farther pixels will influence each other if their colors are close
        """
        self.neigh = neighbourhood
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply_filter(self, image):
        """
        Applies bilateral filtering on the image
        :param image: image as numpy array
        :return: image as numpy array
        """
        return cv2.bilateralFilter(image, self.neigh, self.sigma_color, self.sigma_space)

    def mean_color_diffs(self, image, bilateraled_image=None):
        """
        Counts mean color differences within all colors.
        :param image: original image as matrix cv2_image
        :param bilateraled_image: image as matrix after bilateral filter
        :return: mean difference (any number)
        """
        if bilateraled_image is not None:
            im_bil = bilateraled_image.reshape((-1, 3))
        else:
            im_bil = self.apply_filter(image).reshape((-1, 3))
        im = image.reshape((-1, 3))

        diff = np.mean(im - im_bil, axis=1)
        return np.mean(diff)

    def n_color_diff(self, image, bilateraled_image=None):
        """
        Counts normalized difference between unique colors in image before and after filter.
        :param image: original image as matrix cv2_image
        :param bilateraled_image: image as matrix after bilateral filter
        :return: normalized number of unique colors differences (from 0 to 1)
        """
        if bilateraled_image is not None:
            im_bil = bilateraled_image.reshape((-1, 3))
        else:
            im_bil = self.apply_filter(image).reshape((-1, 3))
        im = image.reshape((-1, 3))
        n_pix = im.shape[0]

        n_colors_im = np.unique(im, axis=0).shape[0]
        n_colors_im_bil = np.unique(im_bil, axis=0).shape[0]

        n_colors_diff = n_colors_im - n_colors_im_bil

        factor = n_colors_diff / n_pix
        return factor

    def _transform_to_hsv(self, image):
        return mpl.colors.rgb_to_hsv(image)

    def h_from_hsv_differences(self, image, bil_image):
        image_hsv = self._transform_to_hsv(image)
        quant_hsv = self._transform_to_hsv(bil_image)

        h_diff = np.mean(image_hsv[:, :, 0] - quant_hsv[:, :, 0])
        return h_diff