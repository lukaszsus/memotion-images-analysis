import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from data_loader.utils import load_image_by_cv2
from feature_extraction.color_counter import ColorCounter


class HoughLines:
    def __init__(self):
        pass

    def get_image_with_lines(self, img, min_line_len=150, plot=False):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        if plot:
            # plt.figure(figsize=(5, 10))
            plt.imshow(edges, cmap='gray')
            plt.axis('off')
            plt.show()

        image = img.copy()

        lines = cv2.HoughLines(edges, 1, np.pi / 180, min_line_len)
        # print(f'Lines: {len(lines)}')
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
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 10)

            if plot:
                # plt.figure(figsize=(5, 10))
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.show()

        return image, int(np.sum(edges) / 255) / (edges.shape[0] * edges.shape[1])

    def get_bounding_boxes(self, img, plot=False):
        size = list(img.shape[:2])
        size.reverse()  # get a [width,height] list

        infos = {
            'size': size,
            'panels': []
        }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        tmin = 220
        tmax = 255
        ret, thresh = cv2.threshold(gray, tmin, tmax, cv2.THRESH_BINARY_INV)

        # plt.imshow(thresh)
        # plt.show()

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get panels out of contours
        for contour in contours:
            arclength = cv2.arcLength(contour, True)

            epsilon = 0.01 * arclength
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # print('\nApprox:', approx.shape, approx)
            x, y, w, h = cv2.boundingRect(approx)

            # exclude very small panels
            if w < infos['size'][0] / 15 or h < infos['size'][1] / 15 or approx.shape[0] > 4:
                # print('...panel excluded\n')
                continue

            contourSize = int(sum(infos['size']) / 2 * 0.01)
            cv2.drawContours(img, [approx], 0, (255, 0, 0), contourSize)

            panel = [x, y, w, h]
            infos['panels'].append(panel)

        if len(infos['panels']) == 0:
            infos['panels'].append([0, 0, infos['size'][0], infos['size'][1]])

        for panel in infos['panels']:
            x, y, w, h = panel
            panel = {
                'x': x,
                'y': y,
                'w': w,
                'h': h
            }

        if plot:
            # plt.figure(figsize=(5, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.show()

        return infos['panels']


if __name__ == "__main__":
    from time import time
    file_names = ['drake', 'cartoon_3', 'cat', 'kermit', 'none_5', 'stats']
    labels = [2, 3, 2, 1, 1, 3]

    for i, file_name in enumerate(file_names):
        # file_name = 'drake.jpg'
        file_path = os.path.join("base_dataset", "segmentation", file_name + '.jpg')
        loaded_image = load_image_by_cv2(file_path)
        x, y, _ = loaded_image.shape
        cc = ColorCounter()

        print(x, y)
        # print(cc.norm_color_count(loaded_image))
        # print(cc.norm_color_count_without_white_and_black(loaded_image))

        hl = HoughLines()
        lines = [120, 150, 180, 200, 250, 300, 350, 400, 450, 500]
        # lines = [100]
        # lines = np.array(np.linspace(int(upper_border/3), int(upper_border/2), 10), dtype=np.int32)
        print('\n\n-------------------\n', file_name.upper())
        for l in lines:
            print(f'\nMinimum no of points: {l}')
            im, norm_edges = hl.get_image_with_lines(loaded_image.copy(), l, plot=False)
            print(f'Normalized edges: {round(norm_edges, 4)}')
            hl.get_bounding_boxes(im, plot=False)
