"""
Models
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


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