import os
import pandas as pd

from src.exception import CustomError
from src.logger import logging
from src.utils import load_pickle_file, load_data

from src.components.data_transformation import DataTransformation
from src.components.model_train import ModelTrain
from sklearn.preprocessing import LabelEncoder

class TrainPipeline:
      def __init__(self):
            self.train_data_path = os.path.join('artifacts', 'train.csv')
            self.test_data_path = os.path.join('artifacts', 'test.csv')

      def execute_pipeline(self):
            try:
                  logging.info("Loading training and test datasets.")
                  train_data = load_data(self.train_data_path)
                  test_data = load_data(self.test_data_path)

                  X_train = train_data.drop(columns=['Species'])
                  y_train = train_data['Species']
                  X_test = test_data.drop(columns=['Species'])
                  y_test = test_data['Species']

                  # Data Transformation
                  data_transformation = DataTransformation()
                  preprocessor = data_transformation.initiate_data_transformation(X_train)
                  X_train_transformed = preprocessor.fit_transform(X_train)
                  X_test_transformed = preprocessor.transform(X_test)       


                  # Encode the target labels
                  label_encoder = LabelEncoder()
                  y_train_encoded = label_encoder.fit_transform(y_train)
                  y_test_encoded = label_encoder.transform(y_test)

                  # Model Training
                  model_train = ModelTrain()
                  logging.info("Starting model training.")
                  model_train.initiate_model_train(X_train_transformed, y_train_encoded, X_test_transformed, y_test_encoded)

                  logging.info("Training pipeline executed successfully.")

            except Exception as e:
                  raise CustomError(f"Error in execute_pipeline: {str(e)}")
if __name__=='__main__':
      train_pipeline = TrainPipeline()
      train_pipeline.execute_pipeline()