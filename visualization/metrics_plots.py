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


def load_datasets_metrics(filename):
    path = os.path.join(DATA_PATH, f'results/{filename}')
    with open(path, "rb") as fin:
        df_metrics, df_y, confusion_matrices = pkl.load(fin)
    return df_metrics, df_y, confusion_matrices


def load_filenames(data_type):
    LABELS = ('cartoon', 'painting', 'photo', 'text')
    image_names = []

    for label_name in LABELS:
        path = os.path.join(DATA_PATH, label_name)
        data_path = os.path.join(path, data_type)
        img_names = [f for f in glob.glob(f'{data_path}/*') if os.path.isfile(f)]
        image_names += img_names

    return image_names, LABELS


class MetricPlotter():

    def __init__(self):
        pass

    def plot_df_metrics(self, df_metrics):
        sns.set_palette("nipy_spectral")

        df = df_metrics.set_index(['Classifier'])
        df.index = df.index.str.wrap(12)
        df.plot.bar(rot=0)
        plt.show()

    def plot_wrongly_classified_pictures(self, df_y, classifier):
        picture_names, label_names = load_filenames('pics')
        df_differ = df_y.loc[df_y['True labels'] != df_y[classifier]][['True labels', classifier]]
        df_differ['Filename'] = np.take(picture_names, df_differ.index.values)

        for i, pred_label in enumerate(label_names):
            pred_as_label = df_differ[df_differ[classifier] == i].values

            fig, ax = plt.subplots(1, 5, figsize=(20, 8))
            fig.suptitle(f'Predicted label: {pred_label}', fontsize=36, fontweight='bold')
            pictures = np.random.permutation(pred_as_label)
            for j in range(5):
                img = cv2.imread(pictures[j][-1])
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax[j].imshow(image)
                ax[j].set_title(f'real: {label_names[pictures[j][0]]}', fontdict={'fontsize': 24})
                ax[j].grid(False)
                ax[j].axis('off')

            plt.show()


if __name__ == "__main__":
    filename = 'metrics_full_for-pics_bilateral_30_50_new-pics_bilateral_30_50-pics_bilateral_50_40.pkl'
    df_metrics, df_y, confusion_matrices = load_datasets_metrics(filename)

    mp = MetricPlotter()
    mp.plot_df_metrics(df_metrics)

    classifier = 'k Nearest Neighbours'
    mp.plot_wrongly_classified_pictures(df_y, classifier)

