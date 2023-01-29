import argparse
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from models import Model
from preprocess import load_and_preprocess
from utils import (
    calculate_results,
    create_tensorboard_callback,
    create_model_checkpoint,
    make_train_test_splits,
    make_windows,
    make_preds,
    evaluate_preds,
    plot_time_series)

SAVE_DIR = "model_logs"

def Experiments(dataset_path: str):
    # Parse dates and set date column to index
    df = pd.read_csv(dataset_path, 
                    parse_dates=["Date"], 
                    index_col=["Date"]) # parse the date column (tell pandas column 1 is a datetime)
    bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    timesteps = bitcoin_prices.index.to_numpy()
    prices = bitcoin_prices["Price"].to_numpy()

    # plt.figure(figsize=(10, 7))
    # plot_time_series(timesteps=timesteps, values=prices)
    # plt.xlabel("Date")
    # plt.ylabel("BTC Price");
    # plt.show()

    X_train, X_test, y_train, y_test = make_train_test_splits(timesteps, prices, 0.2)

    # plt.figure(figsize=(10, 7))
    # plot_time_series(timesteps=X_train, values=y_train, label="Train data")
    # plot_time_series(timesteps=X_test, values=y_test, label="Test data")
    # plt.show()


    models = Model()

    full_windows, full_labels = make_windows(prices, window_size=7, horizon=1)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    # Fit model
    model_name = "model_1_dense"
    model_1 = models.Model_1(horizon=1, name=model_name)
    model_1.fit(x=train_windows, # train windows of 7 timesteps of Bitcoin prices
                y=train_labels, # horizon value of 1 (using the previous 7 timesteps to predict next day)
                epochs=100,
                verbose=1,
                batch_size=128,
                validation_data=(test_windows, test_labels),
                callbacks=[create_model_checkpoint(model_name=model_name)]) # create ModelCheckpoint callback to save best model

    # Evaluate model on test data
    # Load in saved best performing model_1 and evaluate on test data
    model_1 = tf.keras.models.load_model("model_experiments/" + model_name)
    model_1.evaluate(test_windows, test_labels)

    model_1_preds = make_preds(model_1, test_windows)
    model_1_results = evaluate_preds(y_true=tf.squeeze(test_labels), # reduce to right shape
                                 y_pred=model_1_preds)
    print("\n------------\nExperiment 1 results: ", model_1_results)


    full_windows, full_labels = make_windows(prices, window_size=30, horizon=1)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    # Fit model
    model_name = "model_2_dense"
    model_2 = models.Model_1(horizon=1, name=model_name)
    model_2.fit(x=train_windows, # train windows of 7 timesteps of Bitcoin prices
                y=train_labels, # horizon value of 1 (using the previous 7 timesteps to predict next day)
                epochs=100,
                verbose=1,
                batch_size=128,
                validation_data=(test_windows, test_labels),
                callbacks=[create_model_checkpoint(model_name=model_name)]) # create ModelCheckpoint callback to save best model

    # Evaluate model on test data
    # Load in saved best performing model and evaluate on test data
    model_2 = tf.keras.models.load_model("model_experiments/" + model_name)
    model_2.evaluate(test_windows, test_labels)

    model_2_preds = make_preds(model_2, test_windows)
    model_2_results = evaluate_preds(y_true=tf.squeeze(test_labels), # reduce to right shape
                                 y_pred=model_2_preds)
    print("\n------------\nExperiment 2 results: ", model_2_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data", required=True, default="dataset",
                        help="Path to dataset dir", type=str)
    parser.add_argument('-e', "--exp", required=False, default=False,
                        help="Enable experiments", type=bool)
    args = parser.parse_args()
    dataset_path = args.data
    enable_experiments = args.exp

    if enable_experiments:
        Experiments(dataset_path)