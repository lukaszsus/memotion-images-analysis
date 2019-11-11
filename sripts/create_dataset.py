import os

from feature_selection.dataset_creator import DatasetCreator
from settings import DATA_PATH


def create_dataset():
    """
    Example of creating simple test dataset.
    :return:
    """
    creator = DatasetCreator()
    creator.set_to_scalar_features_extractor()
    ds_path = os.path.join(DATA_PATH, "test_dataset")
    creator.create_dataset_from_dir(ds_path)
    file_path = os.path.join(DATA_PATH, "test_dataset.csv")
    creator.save(file_path)


def create_dataset_hsv_var():
    """
    Example of creating simple test dataset.
    :return:
    """
    creator = DatasetCreator()
    creator.set_to_scalar_features_extractor()
    ds_path = os.path.join(DATA_PATH, "test_dataset")
    creator.create_dataset_from_dir(ds_path)
    file_path = os.path.join(DATA_PATH, "test_dataset_hsv_var.csv")
    creator.save(file_path)


if __name__ == '__main__':
    # create_dataset()
    create_dataset_hsv_var()
