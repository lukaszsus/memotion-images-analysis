import colorsys

import matplotlib
import numpy as np


class HsvAnalyser():
    def __init__(self, im: np.array=None):
        self.n_bins = [15, 20, 25]      # numbers of bins used to create histogram
        self.im = None
        if im is not None:
            self.set_image(im)

    def set_image_and_convert_to_hsv(self, im: np.array):
        self.im_hsv = matplotlib.colors.rgb_to_hsv(im)
        self.cos_hues = np.cos(self.im_hsv[:, :, 0])
        self.saturation = self.im_hsv[:, :, 1]
        self.value = self.im_hsv[:, :, 2] / 255.0     # normalization

    def hsv_var(self, image=None, filter_size=(5, 5)):
        """
        Count variance of colors in HSV model. It counts local variance separately for every dimension in HSV.
        :param image: image in RGB, self.values scale 0-255, numpy
        :param filter_size: filter size to determine meaning of 'local'
        :return: standard stats (mean, percentiles) of variances in H, S and V spaces
        """
        if image is not None:
            self.set_image_and_convert_to_hsv(image)
        variances = list()
        ver_step, hor_step = filter_size

        im = np.stack([self.cos_hues, self.saturation, self.value], axis=2)
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

    def saturation_distribution(self, image=None, n_bins = None):
        """
        It prepares self.saturation distribution. It makes them for different numbers of bins,
        defined in n_bins.
        :param image: image to process
        :param n_bins: List of n_bins to make histogram of; if None it takes default [15, 20, 25].
        :return:
        """
        if image is not None:
            self.set_image_and_convert_to_hsv(image)
        n_bins = self.__parse_n_bins(n_bins)
        distributions = list()
        for n_bin in n_bins:
            dist, _ = np.histogram(self.saturation.flatten(), bins=n_bin)
            # Normalize it manually because I want distribution to sum up to 1.
            # Arguments 'density' and 'normed' in np.histogram does not work like that.
            dist = dist / np.sum(dist)
            distributions.extend(dist)
        return np.array(distributions)

    def sat_value_distribution(self, image=None, n_bin=None):
        """
        It prepares 2d distribution of self.saturation and self.value. For examined examples, self.saturation histograms
        came out strange, with a huge number of unsaturated pixels. Probably because of white and black
        background and text in memes. I decided to create second method sat_self.value_distribution to distinguish them.
        I prepares 2 dimensional histogram.
        It counts histogram only for one number of bins because if makes a fairly long feature vector.
        :param image:
        :param n_bin: Number of bins to make histogram of; if None it takes default 20.
        :return:
        """
        if image is not None:
            self.set_image_and_convert_to_hsv(image)
        n_bin = 20 if n_bin is None else n_bin
        distribution, _s, _v = np.histogram2d(self.saturation.flatten(), self.value.flatten(), bins=(n_bin, n_bin))  # density makes normalization
        # Normalize it manually because I want distribution to sum up to 1.
        # Arguments 'density' and 'normed' in np.histogram does not work like that.
        distribution = distribution / np.sum(distribution)
        return np.array(distribution).flatten()

    def __get_stat_metrics(self, values):
        stats = list()
        stats.append(np.mean(values))
        stats.append(np.percentile(values, 10))
        stats.append(np.percentile(values, 25))
        stats.append(np.median(values))
        stats.append(np.percentile(values, 75))
        stats.append(np.percentile(values, 90))
        return np.array(stats)

    def __parse_n_bins(self, n_bins, default = None):
        if default is None:
            default = self.n_bins
        elif type(default) != list:
            default = [default]
        if n_bins is None:
            n_bins = default
        elif type(n_bins) != list:
            n_bins = [n_bins]
        return n_bins