import argparse
import matplotlib.pyplot as plt
import numpy as np
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


def preprocess_for_block_reward(bitcoin_prices):
    # Block reward values
    block_reward_1 = 50 # 3 January 2009 (2009-01-03) - this block reward isn't in our dataset (it starts from 01 October 2013)
    block_reward_2 = 25 # 28 November 2012 
    block_reward_3 = 12.5 # 9 July 2016
    block_reward_4 = 6.25 # 11 May 2020

    # Block reward dates (datetime form of the above date stamps)
    block_reward_2_datetime = np.datetime64("2012-11-28")
    block_reward_3_datetime = np.datetime64("2016-07-09")
    block_reward_4_datetime = np.datetime64("2020-05-11")

    # Get date indexes for when to add in different block dates
    block_reward_2_days = (block_reward_3_datetime - bitcoin_prices.index[0]).days
    block_reward_3_days = (block_reward_4_datetime - bitcoin_prices.index[0]).days

    # Add block_reward column
    bitcoin_prices_block = bitcoin_prices.copy()
    bitcoin_prices_block["block_reward"] = None

    # Set values of block_reward column (it's the last column hence -1 indexing on iloc)
    bitcoin_prices_block.iloc[:block_reward_2_days, -1] = block_reward_2
    bitcoin_prices_block.iloc[block_reward_2_days:block_reward_3_days, -1] = block_reward_3
    bitcoin_prices_block.iloc[block_reward_3_days:, -1] = block_reward_4

    return bitcoin_prices_block


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


    full_windows, full_labels = make_windows(prices, window_size=30, horizon=7)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    # Fit model
    model_name = "model_3_dense"
    model_3 = models.Model_1(horizon=7, name=model_name)
    model_3.fit(x=train_windows, # train windows of 30 timesteps of Bitcoin prices
                y=train_labels, # horizon value of 7 (using the previous 30 timesteps to predict next 7 day)
                epochs=100,
                verbose=1,
                batch_size=128,
                validation_data=(test_windows, test_labels),
                callbacks=[create_model_checkpoint(model_name=model_name)]) # create ModelCheckpoint callback to save best model

    # Evaluate model on test data
    # Load in saved best performing model and evaluate on test data
    model_3 = tf.keras.models.load_model("model_experiments/" + model_name)
    model_3.evaluate(test_windows, test_labels)

    model_3_preds = make_preds(model_3, test_windows)
    model_3_results = evaluate_preds(y_true=tf.squeeze(test_labels), # reduce to right shape
                                 y_pred=model_3_preds)
    print("\n------------\nExperiment 3 results: ", model_3_results)


    full_windows, full_labels = make_windows(prices, window_size=7, horizon=1)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    # Fit model
    model_name = "model_4_conv1d"
    model_4 = models.Model_2(horizon=1, name=model_name)
    model_4.fit(x=train_windows, # train windows of 7 timesteps of Bitcoin prices
                y=train_labels, # horizon value of 1 (using the previous 7 timesteps to predict next day)
                epochs=100,
                verbose=1,
                batch_size=128,
                validation_data=(test_windows, test_labels),
                callbacks=[create_model_checkpoint(model_name=model_name)]) # create ModelCheckpoint callback to save best model

    # Evaluate model on test data
    # Load in saved best performing model and evaluate on test data
    model_4 = tf.keras.models.load_model("model_experiments/" + model_name)
    model_4.evaluate(test_windows, test_labels)

    model_4_preds = make_preds(model_4, test_windows)
    model_4_results = evaluate_preds(y_true=tf.squeeze(test_labels), # reduce to right shape
                                 y_pred=model_4_preds)
    print("\n------------\nExperiment 4 results: ", model_4_results)


    # Fit model
    model_name = "model_5_lstm"
    model_5 = models.Model_3(window_size=7, horizon=1, name=model_name)
    model_5.fit(x=train_windows, # train windows of 7 timesteps of Bitcoin prices
                y=train_labels, # horizon value of 1 (using the previous 7 timesteps to predict next day)
                epochs=100,
                verbose=1,
                batch_size=128,
                validation_data=(test_windows, test_labels),
                callbacks=[create_model_checkpoint(model_name=model_name)]) # create ModelCheckpoint callback to save best model

    # Evaluate model on test data
    # Load in saved best performing model and evaluate on test data
    model_5 = tf.keras.models.load_model("model_experiments/" + model_name)
    model_5.evaluate(test_windows, test_labels)

    model_5_preds = make_preds(model_5, test_windows)
    model_5_results = evaluate_preds(y_true=tf.squeeze(test_labels), # reduce to right shape
                                 y_pred=model_5_preds)
    print("\n------------\nExperiment 5 results: ", model_5_results)


    # Add block reward as data to dataframe
    # Change the univariate to a multivariate data
    bitcoin_prices_block = preprocess_for_block_reward(bitcoin_prices)

    # Make a copy of the Bitcoin historical data with block reward feature
    bitcoin_prices_windowed = bitcoin_prices_block.copy()

    horizon = 1
    window_size = 7
    # Add windowed columns
    for i in range(window_size): # Shift values for each step in WINDOW_SIZE
        bitcoin_prices_windowed[f"Price+{i+1}"] = bitcoin_prices_windowed["Price"].shift(periods=i+1)

    # Create train and test sets. Remove the NaN's and convert to float32 dtype
    X = bitcoin_prices_windowed.dropna().drop("Price", axis=1).astype(np.float32) 
    y = bitcoin_prices_windowed.dropna()["Price"].astype(np.float32)

    X_train, X_test, y_train, y_test = make_train_test_splits(X, y, 0.2)

    # Fit model
    model_name = "model_6_dense_multivariate"
    model_6 = models.Model_1(horizon=horizon, name=model_name)
    model_6.fit(x=X_train, # train windows of 7 timesteps of Bitcoin prices
                y=y_train, # horizon value of 1 (using the previous 7 timesteps to predict next day)
                epochs=100,
                verbose=1,
                batch_size=128,
                validation_data=(X_test, y_test),
                callbacks=[create_model_checkpoint(model_name=model_name)]) # create ModelCheckpoint callback to save best model

    # Evaluate model on test data
    # Load in saved best performing model_1 and evaluate on test data
    model_6 = tf.keras.models.load_model("model_experiments/" + model_name)
    model_6.evaluate(X_test, y_test)

    model_6_preds = make_preds(model_6, X_test)
    model_6_results = evaluate_preds(y_true=tf.squeeze(y_test),
                                 y_pred=model_6_preds)
    print("\n------------\nExperiment 6 results: ", model_6_results)



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