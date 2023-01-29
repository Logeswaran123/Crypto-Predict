import argparse
import tensorflow as tf

from models import Model
from preprocess import load_and_preprocess
from utils import calculate_results, create_tensorboard_callback

SAVE_DIR = "model_logs"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data", required=True, default="dataset",
                        help="Path to dataset dir", type=str)
    parser.add_argument('-e', "--exp", required=False, default=False,
                        help="Enable experiments", type=bool)
    args = parser.parse_args()
    dataset_path = args.data
    enable_experiments = args.exp