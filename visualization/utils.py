import os

from settings import PROJECT_PATH


def create_results_dir():
    """
    It creates directory results and all required subdirectories unless they exist.
    :return:
    """
    main_dir = "results"
    subdirs = ["plots", "tables", "models"]
    main_dir_path = os.path.join(PROJECT_PATH, main_dir)
    if not os.path.isdir(main_dir_path):
        os.mkdir(main_dir_path)
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir_path, subdir)
        if not os.path.isdir(subdir_path):
            os.mkdir(subdir_path)
