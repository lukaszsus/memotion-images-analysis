#!/usr/bin/python3
# coding: utf-8
import os
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import f1_score

from deep_learning.dataset_loader import create_train_test_ds_generator, INPUT_SIZE, CLASS_NAMES
from deep_learning.utils import plot_acc_loss, prepare_and_plot_confusion_matrix, save_summary, create_dirs, \
    DEEP_LEARNING_PARENT_DIR
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from deep_learning.vgg_model_2 import VggModel2

"""
That's for GPU training and maintaining one session and nice cuda lib loading.
"""
config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.75
session = InteractiveSession(config=config)


def do_experiments():
    models_params = list()

    models_cls = [VggModel2]
    num_epochs = [7]
    num_epochs_fine_tuning = [3]
    batch_sizes = [32]
    optimizers_cls = ['Adam']
    learning_rates = [0.001]  # [0.0001, 0.001]
    dense_sizes_list = [(4096, 4)]  # , (256, 4)]
    dropouts = [0.2]  # [0.0, 0.2, 0.5]

    i = 0
    for model_cls in models_cls:
        for dense_sizes in dense_sizes_list:
            for dropout in dropouts:
                for epochs in num_epochs:
                    for fine_tuning_epochs in num_epochs_fine_tuning:
                        for batch_size in batch_sizes:
                            for optimizer_cls in optimizers_cls:
                                for learning_rate in learning_rates:
                                    row = {"model_name": "model{}".format(i),
                                           "model_cls": model_cls,
                                           "dense_sizes": dense_sizes,
                                           "dropout": dropout,
                                           "epochs": epochs,
                                           "fine_tuning_epochs": fine_tuning_epochs,
                                           "batch_size": batch_size,
                                           "optimizer_cls": optimizer_cls,
                                           "learning_rate": learning_rate}
                                    models_params.append(row)
                                    i += 1

    columns = ["model_name", "model_cls",
               "dense_sizes",
               "dropout",
               "epochs",
               "fine_tuning_epochs",
               "batch_size",
               "optimizer_cls",
               "learning_rate",
               # "epoch_time", "time",
               "accuracy", "f1_score"]
    results = pd.DataFrame(columns=columns)

    dt = datetime.now().strftime('%Y-%m-%d-t%H-%M')
    for model_params in models_params:
        train_dataset, test_dataset = create_train_test_ds_generator(batch_size=model_params["batch_size"])

        optimizer = tf.keras.optimizers.get(model_params["optimizer_cls"]).from_config(
            {"learning_rate": model_params["learning_rate"]})
        model = model_params["model_cls"](height=INPUT_SIZE["height"], width=INPUT_SIZE["width"],
                                          num_channels=INPUT_SIZE["num_channels"],
                                          dense_sizes=model_params["dense_sizes"],
                                          dropout=model_params["dropout"])
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        history = model.fit_generator(generator=train_dataset, validation_data=test_dataset,
                                      epochs=model_params["epochs"])

        # fine tuning
        model.base_model.trainable = True
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        history = model.fit_generator(generator=train_dataset, validation_data=test_dataset,
                                      epochs=model_params["fine_tuning_epochs"])

        history = history.history
        history["train_acc"] = history["sparse_categorical_accuracy"]
        history["test_acc"] = history["val_sparse_categorical_accuracy"]
        history["train_loss"] = history["loss"]
        history["test_loss"] = history["val_loss"]

        predictions = list()
        y_test = list()
        for images, labels in test_dataset:
            predictions.extend(model.predict(images).numpy().tolist())
            y_test.extend(labels.numpy().tolist())

        f1score = f1_score(y_test, predictions, average='macro') * 100

        params = model_params.copy()
        row = {"model_name": params["model_name"],
               "model_cls": params["model_cls"].__name__,
               "dense_sizes": params["dense_sizes"],
               "dropout": params["dropout"],
               "epochs": params["epochs"],
               "fine_tuning_epochs": params["fine_tuning_epochs"],
               "batch_size": params["batch_size"],
               "optimizer_cls": params["optimizer_cls"],
               "learning_rate": params["learning_rate"],
               # "epoch_time": history["epoch_time"],
               # "time": history["elapsed"],
               "accuracy": history["test_acc"][-1],
               "f1_score": f1score}
        print(row)
        row = pd.DataFrame([row], columns=columns)
        results = results.append(row, ignore_index=True)

        # savings
        dirpath = os.path.join(DEEP_LEARNING_PARENT_DIR, "tables")
        results.to_csv("{}/results-{}.csv".format(dirpath, dt))
        plot_acc_loss(history, params["model_name"])
        prepare_and_plot_confusion_matrix(params["model_name"], predictions, y_test, CLASS_NAMES)
        dirpath = os.path.join(DEEP_LEARNING_PARENT_DIR, "models")
        model.save_weights('{}/{}-{}.h5'.format(
            dirpath, params["model_name"], dt))
        dirpath = os.path.join(DEEP_LEARNING_PARENT_DIR, "models_summaries")
        save_summary(model, dirpath)

        del model
        del train_dataset
        del test_dataset


if __name__ == '__main__':
    create_dirs()
    do_experiments()
