import os
from unittest import TestCase

from feature_selection.dataset_creator import DatasetCreator
from settings import DATA_PATH


class TestDatasetCreator(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.creator = DatasetCreator()

    def test_create_dataset_from_dir(self):
        """
        This test assumes that you have a test_dataset dir in DATA_PATH.
        Directory test_dataset should have images divided into subdirectories - one for every class.
        :return:
        """
        self.creator.set_to_scalar_features_extractor()
        ds_path = os.path.join(DATA_PATH, "test_dataset")
        self.creator.create_dataset_from_dir(ds_path)
        self.assertGreater(len(self.creator.dataset), 0)
        self.assertEqual(len(self.creator.dataset.columns), 10)

    def test_save(self):
        self.test_create_dataset_from_dir()
        file_path = os.path.join(DATA_PATH, "test_dataset.csv")
        self.creator.save(file_path)
        self.assertTrue(os.path.isfile(file_path))
