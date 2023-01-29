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