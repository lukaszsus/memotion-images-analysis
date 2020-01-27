import tensorflow as tf
import time
from tqdm import tqdm


class VggModel(tf.keras.Model):
    def __init__(self, height, width, num_channels, dense_sizes, dropout, loss_object=None, optimizer=None):
        """Inits the class."""
        super().__init__()
        self.base_model = None
        self._init_base_model(height, width, num_channels)
        self.global_avg_pool_layer = None
        self._init_other_layers()
        self.dense_layers = None
        self._init_dense_layers(dense_sizes, dropout)

        self.train_dataset = None
        self.test_dataset = None
        self.train_loss = None
        self.train_accuracy = None
        self.test_loss = None
        self.test_accuracy = None
        self.history = None

        self.__parse_loss_object(loss_object)
        self.__parse_optimizer(optimizer)

    def _init_base_model(self, height, width, num_channels):
        self.base_model = tf.keras.applications.vgg16.VGG16(input_shape=(height, width, num_channels),
                                                            include_top=False,
                                                            weights='imagenet')
        self.base_model.trainable = False

    def _init_other_layers(self):
        self.global_avg_pool_layer = tf.keras.layers.GlobalAveragePooling2D()

    def _init_dense_layers(self, dense_sizes=None, dropout=0.0):
        if dense_sizes is None:
            dense_sizes = [4096, 128, 4]
        self.dense_layers = []
        for i, size in enumerate(dense_sizes):
            if i < len(dense_sizes) - 1:
                self.dense_layers.append(tf.keras.layers.Dense(size, activation=tf.nn.relu,
                                                               kernel_initializer="glorot_normal",
                                                               dtype=tf.float32))
                self.dense_layers.append(tf.keras.layers.Dropout(dropout, dtype=tf.float32))
            else:
                self.dense_layers.append(tf.keras.layers.Dense(size, activation=tf.nn.softmax,
                                                               kernel_initializer="glorot_normal",
                                                               dtype=tf.float32))

    def call(self, inputs, training=False):
        """Makes forward pass of the network."""
        x = self.base_model(inputs)
        x = self.global_avg_pool_layer(x)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        return x

    def fit(self, **kwargs):
        """
        Implements learning loop for the model.
        kwargs can contain optional parameters
        """
        self.train_dataset = kwargs.get('train_dataset')
        self.test_dataset = kwargs.get('test_dataset')
        epochs = kwargs.get('epochs', 10)

        # metrics
        self.__create_metrics_history()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        start_time = time.time()
        for epoch in range(epochs):
            for images, labels in tqdm(self.train_dataset):
                self.train_step(images, labels)

            for x_test, y_test in tqdm(self.test_dataset):
                self.test_step(x_test, y_test)

            template = 'Epoch {}, Loss: {}, Accuracy: {} ' + 'Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result() * 100,
                                  self.test_loss.result(),
                                  self.test_accuracy.result() * 100))

            self.__update_metrics_history()

            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

        elapsed = time.time() - start_time
        epoch_time = elapsed / epochs
        self.history["elapsed"] = elapsed
        self.history["epoch_time"] = epoch_time

        return self.history

    def predict(self, x):
        """Predicts outputs based on inputs (x)."""
        return tf.argmax(self.call(x, training=False), axis=1)

    # tf.function Rzutuje operacje pythonowe na operacje tf
    @tf.function
    def train_step(self, images, labels):
        # what should tf count gradient for
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self(images)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss.update_state(t_loss)
        self.test_accuracy.update_state(labels, predictions)

    def __parse_loss_object(self, loss_object):
        if type(loss_object) == str:
            self.loss_object = tf.keras.losses.get(loss_object)
        elif loss_object is not None:
            self.loss_object = loss_object
        else:
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    def __parse_optimizer(self, optimizer):
        if type(optimizer) == str:
            self.optimizer = tf.keras.optimizers.get(optimizer)
        elif optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = tf.keras.optimizers.Adam()

    def __create_metrics_history(self):
        self.history = {"train_loss": list(),
                        "train_acc": list(),
                        "test_loss": list(),
                        "test_acc": list(),
                        "elapsed": None,
                        "epoch_time": None}

    def __update_metrics_history(self):
        self.history["train_loss"].append(self.train_loss.result().numpy())
        self.history["train_acc"].append(self.train_accuracy.result().numpy() * 100)
        self.history["test_loss"].append(self.test_loss.result().numpy())
        self.history["test_acc"].append(self.test_accuracy.result().numpy() * 100)
