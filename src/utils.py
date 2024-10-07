# src/utils.py

import os
import sys
import pickle
from src.exception import CustomError
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path} successfully.")
        return data
    except Exception as e:
        raise CustomError(f"Failed to load data from {file_path}: {str(e)}", sys)

def save_data(data, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False, header=True)
        logging.info(f"Data saved to {file_path} successfully.")
    except Exception as e:
        raise CustomError(f"Failed to save data to {file_path}: {str(e)}", sys)

def split_data(df, target_column, test_size=0.2, random_state=42):
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into train and test sets with test size {test_size}.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomError(f"Failed to split data: {str(e)}", sys)

def save_pickle_file(file_path, obj):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

        logging.info(f"Saved the file in {file_path}")
    except Exception as e:
        raise CustomError(f"Failed to save file to {file_path}: {str(e)}", sys)

def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)

        logging.info(f"Loaded the file from {file_path}")
        return obj
    except Exception as e:
        raise CustomError(f"Failed to load file from {file_path}: {str(e)}", sys)

# Main function to test the utilities
if __name__ == "__main__":
    logging.info("Testing utils.py")