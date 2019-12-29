import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from data_loader.utils import load_image_by_cv2


class HoughLines:
    def __init__(self):
        pass

    def plot_lines(self, img, min_line_len=150):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        plt.figure(figsize=(5, 10))
        plt.imshow(edges, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()

        image = img.copy()

        lines = cv2.HoughLines(edges, 1, np.pi / 180, min_line_len)
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

        plt.figure(figsize=(5, 10))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

        return image

    def get_bounding_boxes(self, img):
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
            if w < infos['size'][0] / 15 or h < infos['size'][1] / 15:  # or approx.shape[0] > 4:
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

        plt.figure(figsize=(5, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        return infos['panels']


if __name__ == "__main__":
    file_name = 'kermit.jpg'
    file_path = os.path.join("base_dataset", "segmentation", file_name)
    image = load_image_by_cv2(file_path)

    hl = HoughLines()
    image = hl.plot_lines(image, 250)
    hl.get_bounding_boxes(image)