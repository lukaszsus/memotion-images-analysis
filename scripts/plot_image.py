import os

from data_loader.utils import load_image_by_cv2
from settings import DATA_PATH
from visualization.single_image_plotter import SingleImagePlotter

if __name__ == '__main__':
    im_plotter = SingleImagePlotter()
    file_name = 'camera.jpg'
    dir_path = os.path.join(DATA_PATH, "test_dataset/photo")
    # file_name = 'pepe.png'
    # dir_path = os.path.join(DATA_PATH, "test_dataset/cartoon")
    im_photo = load_image_by_cv2(os.path.join(dir_path, file_name))
    im_plotter.plot_image(im_photo)
    im_plotter.plot_image_hist(im_photo)
    im_plotter.plot_image_hist_rgb(im_photo)