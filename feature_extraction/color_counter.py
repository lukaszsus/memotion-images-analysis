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
