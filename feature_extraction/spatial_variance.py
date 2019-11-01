import colorsys

import matplotlib
import numpy as np


class SpatialVariance():
    def __init__(self):
        pass

    def angle_cos_var(self, image, filter_size):
        """
        Count variance of colors in HSV model. It counts local variance separately for every dimension in HSV.
        :param image: image in RGB, values scale 0-255
        :param filter_size: filter size to determine meaning of 'local'
        :return: standard stats (mean, q25, median, q75) of variances in H, S and V spaces
        """
        variances = list()
        ver_step, hor_step = filter_size
        im_hsv = matplotlib.colors.rgb_to_hsv(image)

        cos_hues = np.cos(im_hsv[:, :, 0])
        saturation = im_hsv[:, :, 1]
        value = im_hsv[:, :, 2] / 255.0     # normalization

        im = np.stack([cos_hues, saturation, value], axis=2)
        for i in range(im.shape[0] - ver_step):
            for j in range(im.shape[1] - hor_step):
                local_im = im[i:i + ver_step, j:j + hor_step, :]
                variance = np.var(local_im, axis=(0, 1))
                variances.append(variance)
        variances = np.array(variances)
        hue_stats = self.__get_stat_metrics(variances[:, 0])
        sat_stats = self.__get_stat_metrics(variances[:, 1])
        value_stats = self.__get_stat_metrics(variances[:, 2])
        return np.concatenate([hue_stats, sat_stats, value_stats])


    def __get_stat_metrics(self, values):
        mean = np.mean(values)
        q25 = np.percentile(values, 25)
        median = np.median(values)
        q75 = np.percentile(values, 75)
        return np.array([mean, q25, median, q75])