import tensorflow as tf
import tensorflow.keras as keras

class TestModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_layer = keras.layers.Dense(5)
        self.output_layer = keras.layers.Dense(5)

    def call(self, data):
        x = self.dense_layer(data)
        x = self.output_layer(x)

        return x
