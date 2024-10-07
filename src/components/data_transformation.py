import os
import sys
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from src.exception import CustomError
from src.logger import logging
from src.utils import save_pickle_file

class DataTransformationConfig:
      def __init__(self):
            self.preprocess_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
      def __init__(self):
            self.data_transformation = DataTransformationConfig()

      def initiate_data_transformation(self, X_train):
            try:
                  logging.info("Data transformation start.")

                  numerical_columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

                  num_pipeline= Pipeline(steps=[
                        ('imputers', SimpleImputer(strategy='mean')),
                        ('scalers', StandardScaler())
                  ])

                  preprocessor = ColumnTransformer(transformers=[
                        ('num_pipeline', num_pipeline, numerical_columns)
                  ])
                  logging.info("Data clean and make for transformation.")

                  preprocessor.fit(X_train)

                  save_pickle_file(self.data_transformation.preprocess_file_path,preprocessor)
                  logging.info("save the preprocessor file.")

                  return preprocessor

            except Exception as e:
                  raise CustomError(str(e), sys)

if __name__=='__main__':
      train_data = pd.read_csv(os.path.join('artifacts', 'train.csv'))
      X_train = train_data.drop(columns=['Species'])
      data_transformation=DataTransformation()
      preprocessor = data_transformation.initiate_data_transformation(X_train)
