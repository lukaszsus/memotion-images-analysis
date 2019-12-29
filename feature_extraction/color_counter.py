import numpy as np


class ColorCounter():
    def __init__(self):
        pass

    def norm_color_count(self, image):
        """
        Counts number of colors divided by number of pixels in picture.
        :return:
        """
        h, w, d = image.shape
        n_pix = h * w
        im = image.reshape((-1, 3))
        n_colors = np.unique(im, axis=0).shape[0]
        factor = n_colors / n_pix
        return factor

    def norm_color_count_without_white_and_black(self, image):
        """
        Counts number of colors divided by number of pixels in picture
        excluding pixels that are white and black - first and last color.
        """
        im = image.reshape((-1, 3))
        colors, counts = np.unique(im, axis=0, return_counts=True)
        n_pix = np.sum(counts[1:-1])
        n_colors = colors.shape[0] - 2  # without first and last
        factor = n_colors / n_pix
        return factor