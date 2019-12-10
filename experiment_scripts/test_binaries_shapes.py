"""
Scripts test if created binaries have a expected shapes.
"""
import os
import pickle

import numpy as np

from settings import DATA_PATH

if __name__ == '__main__':
    dir_path = os.path.join(DATA_PATH, "test_pics_feature_binaries")
    binaries_paths = os.listdir(dir_path)
    for file_name in binaries_paths:
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'rb') as file:
            x = pickle.load(file)
            if type(x) == np.ndarray:
                print("{}: {}".format(file_name, x.shape))
            else:
                print("{}: {}".format(file_name, len(x)))

    with open(os.path.join(dir_path, "pics_paths.pickle"), 'rb') as file:
        x = pickle.load(file)
        print(x)