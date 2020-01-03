import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from data_loader.utils import load_image_by_cv2
from user_settings import DATA_PATH


class HoughLines:
    """
    Class responsible for image segmentation based on Hough Transformation.
    """
    def __init__(self):
        """
        Sets default thresholds values.
        """
        self.min_canny_threshold = 50
        self.max_canny_threshold = 150
        self.min_mask_threshold = 240
        self.max_mask_threshold = 260

    def get_image_with_lines(self, img, min_line_len=None, plot=False, verbose=False):
        """
        Applies Hough Transformation - finds vertical and horizontal lines.
        :param img: image loaded by opencv in RGB
        :param min_line_len: minimum number of collinear points to make a line;
                    if None value will be adjusted automatically - recommended
        :param plot: boolean flag; if True images are plotted
        :return: processed image with white lines | normalised value of edges due to all pixels |
                    value of param for Hough Transform
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.min_canny_threshold, self.max_canny_threshold, apertureSize=3)
        norm_edges = int(np.sum(edges) / 255) / (edges.shape[0] * edges.shape[1])
        x, y = gray.shape

        if plot:
            # plt.figure(figsize=(5, 10))
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            plt.show()

        image = img.copy()

        auto_min_line_len = min(int(norm_edges * min(x, y) * 10), int(0.8 * min(x, y)))
        lines = cv2.HoughLines(edges, 1, np.pi / 180, auto_min_line_len if min_line_len is None else min_line_len)

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
                if abs(abs(x1) - abs(x2)) < 5 and abs(abs(y1) - abs(y2)) < 5 and x1*x2*y1*y2 < 0:
                    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 10)
                elif verbose:
                    print(' non horizontal line detected but not included')

        if plot:
            # plt.figure(figsize=(5, 10))
            plt.imshow(img, cmap='gray')
            plt.title(f'Detected {0 if lines is None else len(lines)} lines')
            plt.axis('off')
            plt.show()

        return image, round(norm_edges, 4), auto_min_line_len

    def get_bounding_boxes(self, img, plot=False, plot_title=''):
        """
        Applies mask on image and finds bounding boxes of pictures on white background.
        :param img: opencv image in RGB
        :param plot: boolean flag
        :return: list of bounding boxes
        """
        size = list(img.shape[:2])
        size.reverse()  # get a [width,height] list

        infos = {
            'size': size,
            'panels': []
        }

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, self.min_mask_threshold, self.max_mask_threshold,
                                    cv2.THRESH_BINARY_INV)

        if plot:
            plt.imshow(thresh)
            plt.axis('off')
            plt.show()

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get panels out of contours
        for contour in contours:
            arclength = cv2.arcLength(contour, True)

            epsilon = 0.01 * arclength
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)

            # exclude to small panels
            if w < infos['size'][0] / 5 or h < infos['size'][1] / 5:
                continue

            approx_points = list(approx.reshape(-1, 2))
            rect_points = list(np.array([[x, y], [x, y+h], [x+w, y], [x+w, y+h]]))
            if self._found_correct_bounding_boxes(rect_points, approx_points):
                contour_size = int(sum(infos['size']) / 2 * 0.01)
                # cv2.drawContours(img, [approx], 0, (0, 255, 0), contour_size)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), contour_size)

                panel = [x, y, w, h]
                infos['panels'].append(panel)

        if len(infos['panels']) == 0:
            infos['panels'].append([0, 0, infos['size'][0], infos['size'][1]])

        if plot:
            # plt.figure(figsize=(5, 10))
            plt.imshow(img)
            plt.title(plot_title)
            plt.axis('off')
            plt.show()

        return infos['panels']

    def _found_correct_bounding_boxes(self, rect_points, approx_points):
        """
        Checks if contour (approx_points) has at least 3 common corners with detected bounding box.
        :param rect_points: real corners (bounding box)
        :param approx_points: all contour points (not necessarily rectangle)
        :return: True if contour has at least 3 real corners
        """
        results = []
        for i, a in enumerate(rect_points):
            results.append([])
            for b in approx_points:
                results[i].append(np.linalg.norm(a - b))
        res = [sorted(r)[0] for r in results]
        return sum(sorted(res)[:-1]) < 15


if __name__ == "__main__":
    file_name = 'tumblr_8'
    file_path = os.path.join(DATA_PATH, "base_dataset", "segmentation", "tests")
    filename = glob.glob(f'{file_path}/{file_name}*')[0]

    loaded_image = load_image_by_cv2(filename)
    x, y, _ = loaded_image.shape

    hl = HoughLines()
    l = 250

    print(f'\nMinimum no of points: {"automatic" if l is None else l}')
    im, norm_edges, auto_min_line_len = hl.get_image_with_lines(loaded_image.copy(), l, plot=True)
    print(f'Normalized edges: {norm_edges}')
    print(f'Automatic value of min_lines: {auto_min_line_len}')

    boxes = hl.get_bounding_boxes(im, plot=True)
    print(f'Bounding boxes: {len(boxes)}')
