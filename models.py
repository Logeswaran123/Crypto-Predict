"""
Models
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# Create NBeatsBlock custom layer
# Refer: https://arxiv.org/pdf/1905.10437.pdf (N-BEATS)
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self,
                input_size: int,
                theta_size: int,
                horizon: int,
                num_hidden_units: int,
                num_hidden_layers: int,
                **kwargs):  # the **kwargs argument takes care of all of the arguments for the parent class (input_shape, trainable, name)
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers

        self.hidden = [tf.keras.layers.Dense(self.num_hidden_units, activation="relu") for _ in range(self.num_hidden_layers)]

        # Output of block is a theta layer with linear activation
        self.theta_layer = tf.keras.layers.Dense(self.theta_size, activation="linear", name="theta")

    def call(self, inputs):
        x = inputs 
        for layer in self.hidden:   # pass inputs through each hidden layer 
            x = layer(x)
        theta = self.theta_layer(x) 

        # Output the backcast and forecast from theta
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast


class Model():
    """
    Class with multiple model architectures
    """
    def __init__(self) -> None:
        pass

    def Baseline(self):
        """
        Create and return a Baseline model
        """
        pass

    def Model_1(self, horizon: int, name: str):
        """
        Create and return a simple Dense Model
        """
        model = tf.keras.Sequential([
                    layers.Dense(128, activation="relu"),
                    layers.Dense(horizon, activation="linear")  # linear activation is the same as having no activation                        
                    ], name=name)

        # Compile model
        model.compile(loss="mae",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["mae"])    # we don't necessarily need this when the loss function is already MAE

        return model

    def Model_2(self, horizon: int, name: str):
        """
        Create and return a Conv1D Model
        """
        model = tf.keras.Sequential([
                    # Create Lambda layer to reshape inputs for Conv1D 3D input shape requirements.
                    layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
                    layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu"), # "causal" padding used for temporal data
                    layers.Dense(horizon)
                    ], name=name)

        # Compile model
        model.compile(loss="mae",
                    optimizer=tf.keras.optimizers.Adam())

        return model

    def Model_3(self, window_size: int, horizon: int, name: str):
        """
        Create and return a LSTM Model
        """
        inputs = layers.Input(shape=(window_size))
        x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)  # Create Lambda layer to reshape inputs for LSTM 3D input shape requirements.
        x = layers.LSTM(128, activation="relu")(x)
        output = layers.Dense(horizon)(x)
        model = tf.keras.Model(inputs=inputs,
                    outputs=output,
                    name=name)

        # Compile model
        model.compile(loss="mae",
                    optimizer=tf.keras.optimizers.Adam())

        return model