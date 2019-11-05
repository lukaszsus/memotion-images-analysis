import numpy as np
import cv2


class BilateralFilter():
    def __init__(self, neighbourhood, sigma_color, sigma_space):
        """
        :param neighbourhood: diameter of each pixel neighborhood that is used during filtering
        :param sigma_color: larger value means that farther colors within the pixel neighborhood will be mixed together,
               resulting in larger areas of semi-equal color
        :param sigma_space: larger value means that farther pixels will influence each other if their colors are close
        """
        self.neigh = neighbourhood
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def _apply_filter(self, image):
        """
        Applies bilateral filtering on the image
        :param image: image as numpy array
        :return: image as numpy array
        """
        return cv2.bilateralFilter(image, self.neigh, self.sigma_color, self.sigma_space)

    def mean_color_diffs(self, image):
        im_bil = self._apply_filter(image).reshape((-1, 3))
        im = image.reshape((-1, 3))

        diff = np.mean(im - im_bil, axis=1)
        return np.mean(diff)

    def n_color_diff(self, image):
        im_bil = self._apply_filter(image).reshape((-1, 3))
        im = image.reshape((-1, 3))
        n_pix = im.shape[0]

        n_colors_im = np.unique(im, axis=0).shape[0]
        n_colors_im_bil = np.unique(im_bil, axis=0).shape[0]

        n_colors_diff = n_colors_im - n_colors_im_bil

        factor = n_colors_diff / n_pix
        return factor