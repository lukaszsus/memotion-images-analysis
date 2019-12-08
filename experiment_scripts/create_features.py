"""
Scripts generates feautures and saves them to pickle files.
One pickle file represents one bunch of features, in general it contains
output of one method from feature generator.

I can be run from console, for example:

python3 experiment_scripts/create_features.py --source-path "base_dataset" --type "pics" --dst-path "pics_feature_binaries"
"""
import argparse
import os
import sys
sys.path.append(os.getcwd())
from settings import DATA_PATH
from feature_selection.feature_extractor import FeatureExtractor


def get_args() -> argparse.Namespace:
    """Parses script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-path',
        required=False,
        help='Name of directory with dataset. It has to be inside DATA_PATH.',
        type=str,
    )
    parser.add_argument(
        '--type',
        required=False,
        type=str,
    )
    parser.add_argument(
        '--dst-path',
        required=False,
        help='Name of directory to save features in. It will be inside DATA_PATH.',
        type=str,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    source_path = os.path.join(DATA_PATH, args.source_path) \
        if args.source_path is not None else os.path.join(DATA_PATH, "test_dataset")
    type = args.type if args.type is not None else "pics"
    dst_path = os.path.join(DATA_PATH, args.dst_path) if args.type is not None else os.path.join(DATA_PATH, "test_pics_feature_binaries")

    feature_extractor = FeatureExtractor()
    feature_extractor.create_features_from_dir(src_path=source_path,
                                               im_type=type,
                                               save_path=dst_path)
    feature_extractor.create_labels_from_dir(src_path=source_path,
                                             im_type=type,
                                             save_path=dst_path)
    feature_extractor.create_path_list_from_dir(src_path=source_path,
                                                im_type=type,
                                                save_path=dst_path)
