"""
Preprocess functions
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess(dataset_path):
    """
    Load and preprocess dataset and return training and validation splits

    Args:
    -----
    dataset_path = Path to dataset directory

    Returns training and validation splits.
    """
    # Get data
    train_df = pd.read_csv(dataset_path + "/train.csv")
    test_df = pd.read_csv(dataset_path + "/test.csv")

    # Shuffle all training data
    train_df_shuffled = train_df.sample(frac=1, random_state=42)

    # Split training data into training and validation sets
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1, # dedicate 10% of samples to validation set
                                                                            random_state=42) # random state for reproducibility
    return (train_sentences, val_sentences, train_labels, val_labels)
