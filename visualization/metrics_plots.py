import glob
import os
import pickle as pkl

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from settings import DATA_PATH
sns.set()


def load_datasets_metrics(filename, default_dir='results/metrics/pics'):
    """
    Loads saved in default directory metrics.
    :param filename: name of file to load
    :return: metrics: dataframe with accuracy and fscore, dataframe with y_preds and confussion matrix
    """
    path = os.path.join(DATA_PATH, f'{default_dir}/{filename}')
    with open(path, "rb") as fin:
        df_metrics, df_y, confusion_matrices = pkl.load(fin)
    return df_metrics, df_y, confusion_matrices


def load_filenames(data_type):
    """
    TEMPORARY xD
    Loads filenames in the same order as where saved (without given any order in fact).
    :param data_type: directory from which to load data - memes or pics
    :return: names of images and label names in right order
    """
    LABELS = ('cartoon', 'painting', 'photo', 'text')
    image_names = []

    for label_name in LABELS:
        path = os.path.join(DATA_PATH, label_name)
        data_path = os.path.join(path, data_type)
        img_names = [f for f in glob.glob(f'{data_path}/*') if os.path.isfile(f)]
        image_names += img_names

    return image_names, LABELS


class MetricPlotter():
    """
    Class that plots some metrics using matplotlib.
    """

    def __init__(self):
        pass

    def plot_df_metrics(self, df_metrics):
        """
        Plots barplot with metrics - accuracy and fscore - for each classifier.
        :param df_metrics: dataframe with those metrics
        """
        sns.set_palette("rainbow_r")

        df = df_metrics.set_index(['Classifier'])
        df.index = df.index.str.wrap(12)
        df.plot.bar(rot=0)
        plt.show()

    def plot_wrongly_classified_pictures(self, df_y, classifier, number_of_images=5):
        """
        Plots given number of wrongly classified images with their true and predicted labels.
        It goes through predicted labels and shows true that differs (not the other way round).
        :param df_y: dataframe with predicted and true labels
        :param classifier: classifier name that will be compared with true labels
        :param number_of_images: number of images to show
        """
        picture_names, label_names = load_filenames('pics')
        df_differ = df_y.loc[df_y['True labels'] != df_y[classifier]][['True labels', classifier]]
        df_differ['Filename'] = np.take(picture_names, df_differ.index.values)

        for i, pred_label in enumerate(label_names):
            pred_as_label = df_differ[df_differ[classifier] == i].values

            fig, ax = plt.subplots(1, number_of_images, figsize=(4*number_of_images, 8))
            fig.suptitle(f'Predicted label: {pred_label}', fontsize=36, fontweight='bold')
            pictures = np.random.permutation(pred_as_label)
            for j in range(min(number_of_images, pictures.shape[0])):
                img = cv2.imread(pictures[j][-1])
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax[j].imshow(image)
                ax[j].set_title(f'real: {label_names[pictures[j][0]]}', fontdict={'fontsize': 24})
                ax[j].grid(False)
                ax[j].axis('off')

            plt.show()


if __name__ == "__main__":
    filename = 'metrics_full_for-pics_hsv_analyser_sat_value_distribution-without_photo.pkl'
    df_metrics, df_y, confusion_matrices = load_datasets_metrics(filename, default_dir='results')

    mp = MetricPlotter()
    mp.plot_df_metrics(df_metrics)

    # classifier = 'Random Forest'
    # mp.plot_wrongly_classified_pictures(df_y, classifier)

