import cv2
import numpy as np
from scipy import ndimage as nd

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel


class GaborFilter():
    def __init__(self, thetas=(0, 0.25, 0.5, 0.75), sigmas=(1, 3), frequencies=(0.05, 0.25)):
        """
        Params are the hyperparameters for bank of gabor filters.

        :param thetas: tuple of theta - theta is the orientation of the normal to the parallel stripes
                        of the Gabor function. Rotation is count as theta * pi
        :param sigmas: tuple of sigma - sigma is the standard deviation of the Gaussian function
                        used in the Gabor filter.
        :param frequencies: tuple of spatial frequency of the harmonic function. Specified in pixels.
        """
        self.kernels = None
        self.thetas = thetas
        self.sigmas = sigmas
        self.frequencies = frequencies
        self._prepare_filter_bank_kernels()

    def apply_filter(self, image):
        """
        Applies Gabor filter.
        :param image: image as numpy array
        :return: numpy vector of features; length of vector equals to len(self.kernels) * 2;
                    2 because of returning mean and standard deviation
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self._compute_feats(gray)

    def _prepare_filter_bank_kernels(self):
        self.kernels = list()
        # theta is rotation
        for theta in self.thetas:
            theta = theta * np.pi
            # kernel standard deviation
            for sigma in self.sigmas:
                # Spatial frequency of the harmonic function. Specified in pixels.
                for frequency in self.frequencies:
                    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                  sigma_x=sigma, sigma_y=sigma))
                    self.kernels.append(kernel)

    def _compute_feats(self, image):
        feats = np.zeros((len(self.kernels) * 2), dtype=np.double)
        for k, kernel in enumerate(self.kernels):
            filtered = nd.convolve(image, kernel, mode='wrap')
            feats[2 * k] = filtered.mean()
            feats[2 * k + 1] = filtered.var()
        return feats