import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from settings import DATA_PATH

INPUT_SIZE = {
    "height": 224,
    "width": 224,
    "num_channels": 3
}
CLASS_NAMES = ["cartoon", "painting", "photo", "text"]

plt.ioff()


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=INPUT_SIZE["num_channels"])
    image = tf.image.resize(image, [INPUT_SIZE["height"], INPUT_SIZE["width"]])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def create_file_paths_for_generator():
    """
    By far only for pics.
    """
    dataset_path = os.path.join(DATA_PATH, "base_dataset")
    filepath_list = list()
    classes_list = list()
    for i in range(len(CLASS_NAMES)):
        y_label = CLASS_NAMES[i]
        class_path = os.path.join(dataset_path, y_label)
        class_path = os.path.join(class_path, "pics")
        filenames = os.listdir(class_path)
        for filename in filenames:
            filepath_list.append(os.path.join(class_path, filename))
            classes_list.append(i)

    return filepath_list, classes_list


def create_ds_generator(files, categories, batch_size):
    """
    Creates dataset from list of files paths and list of their categories.
    """
    path_ds = tf.data.Dataset.from_tensor_slices(files)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(categories, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    ds_generator = image_label_ds.shuffle(buffer_size=1000 * batch_size)
    ds_generator = ds_generator.batch(batch_size)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    ds_generator = ds_generator.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds_generator


def create_train_test_ds_generator(batch_size):
    files, categories = create_file_paths_for_generator()
    train_files, test_files, train_categories, test_categories = train_test_split(files, categories,
                                                                                  test_size=0.1, random_state=42,
                                                                                  stratify=categories)
    train_ds_generator = create_ds_generator(train_files, train_categories, batch_size)
    test_ds_generator = create_ds_generator(test_files, test_categories, batch_size=1)
    return train_ds_generator, test_ds_generator


########################## Witek inspired way to dynamic load data

# def load_images(images):
#     img = cv2.imread(images.numpy().decode('utf-8'))
#     img = cv2.resize(img, (HEIGHT, WIDTH), cv2.INTER_LINEAR)
#     img = preprocess_input(img)
#     return img
#
#
# def data_generator(images, labels):
#     img = tf.convert_to_tensor(
#         tf.py_function(load_images, [images], tf.float32))
#     return img, labels
#
#
# def create_train_test_ds_generator(batch_size):
#     files, categories = create_file_paths_for_generator()
#     train_files, test_files, train_categories, test_categories = train_test_split(files, categories,
#                                                                                   test_size=0.1, random_state=42,
#                                                                                   stratify=categories)
#     train_dataset = tf.data.Dataset.from_tensor_slices(
#         (train_files, train_categories)).map(
#         data_generator).batch(
#         batch_size).prefetch(buffer_size=AUTOTUNE)
#     test_dataset = tf.data.Dataset.from_tensor_slices(
#         (test_files, test_categories)).map(
#         data_generator).batch(
#         batch_size).prefetch(buffer_size=AUTOTUNE)
#
#     return train_dataset, test_dataset

##############################################