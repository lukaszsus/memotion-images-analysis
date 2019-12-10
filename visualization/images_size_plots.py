import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from data_loader.utils import load_image_by_cv2
from settings import DATA_PATH


def avg_img_size(src_path, im_type):
    labels_choice = os.listdir(src_path)
    labels_choice.sort()
    sizes = list()
    x = list()
    y = list()
    labels = list()
    for i in range(len(labels_choice)):
        label = labels_choice[i]
        print("Label: {}".format(label))
        label_path = os.path.join(src_path, label)
        if im_type is not None:
            label_path = os.path.join(label_path, im_type)
        files = os.listdir(label_path)
        files.sort()
        for file_name in tqdm(files):
            file_path = os.path.join(label_path, file_name)
            im = load_image_by_cv2(file_path)
            sizes.append(np.mean(im.shape[0:2]))
            y.append(im.shape[0])
            x.append(im.shape[1])
            labels.append(i)
    return sizes, x, y, labels


if __name__ == '__main__':
    sizes, x, y, labels = avg_img_size(os.path.join(DATA_PATH, "base_dataset"), "pics")
    plt.hist(sizes, bins=20)
    plt.ylabel('Number of images')
    plt.xlabel('Average of width and height')
    plt.savefig(os.path.join(DATA_PATH, "data_analysis/plots/im_sizes_hist.pdf"), bbox_inches='tight')
    plt.show()

    print(np.mean(sizes))
    print(np.percentile(sizes, 10))
    print(np.percentile(sizes, 25))
    print(np.percentile(sizes, 50))
    print(np.percentile(sizes, 75))
    print(np.percentile(sizes, 90))

    l_choice = ["cartoon", "painting", "photo", "text"]
    l = [l_choice[label] for label in labels]

    df = pd.DataFrame(columns=["width", "height", "label"])
    df["width"] = x
    df["height"] = y
    df["label"] = l
    sns.scatterplot(x="width", y="height", hue="label", data=df)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DATA_PATH, "data_analysis/plots/im_sizes_scatter.pdf"), bbox_inches='tight')
    plt.show()