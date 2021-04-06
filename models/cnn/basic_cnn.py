import tensorflow as tf
from tensorflow.keras import layers, models


class BasicCNN(models.Sequential):
    def __init__(self, input_size, classes_count):
        super(BasicCNN, self).__init__()

        # Store attributes
        self.input_size = input_size
        self.classes_count = classes_count

        # CNN layers
        self.conv_1 = layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=input_size
        )
        self.max_pool_1 = layers.MaxPooling2D((2, 2))
        self.conv_2 = layers.Conv2D(64, (3, 3), activation="relu")
        self.max_pool_2 = layers.MaxPooling2D((2, 2))
        self.conv_3 = layers.Conv2D(64, (3, 3), activation="relu")

        # Classification layers
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(64, activation="relu")
        self.classifier = layers.Dense(classes_count, activation="sigmoid")

        # Compile
        self.compile(
            optimizer="rmsprop",
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.AUC(),
                tf.keras.metrics.TruePositives(),
                tf.keras.metrics.FalseNegatives(),
                tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.FalsePositives(),
            ],
        )

    def call(self, inputs):
        # CNN Layers
        x = self.conv_1(inputs)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.max_pool_2(x)
        x = self.conv_3(x)

        # CNN Classification
        x = self.flatten(x)
        x = self.dense_1(x)
        return self.classifier(x)
