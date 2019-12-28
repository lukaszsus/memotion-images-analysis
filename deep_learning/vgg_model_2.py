import tensorflow as tf


class VggModel2(tf.keras.Model):
    """
    The difference between VggModel and VggModel2 is that VggModel2 uses built-in compile and fit methods.
    """
    def __init__(self, height, width, num_channels, dense_sizes, dropout):
        """Inits the class."""
        super().__init__()
        self.base_model = None
        self._init_base_model(height, width, num_channels)
        self.global_avg_pool_layer = None
        self._init_other_layers()
        self.dense_layers = None
        self._init_dense_layers(dense_sizes, dropout)

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
                                                               kernel_initializer="glorot_normal"))
                self.dense_layers.append(tf.keras.layers.Dropout(dropout))
            else:
                self.dense_layers.append(tf.keras.layers.Dense(size, activation=tf.nn.softmax,
                                                               kernel_initializer="glorot_normal"))

    def call(self, inputs, training=False):
        """Makes forward pass of the network."""
        x = self.base_model(inputs)
        x = self.global_avg_pool_layer(x)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        return x

    def predict(self, x):
        """Predicts outputs based on inputs (x)."""
        return tf.argmax(self.call(x, training=False), axis=1)

