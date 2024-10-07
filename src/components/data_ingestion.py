import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomError
from src.logger import logging
from src.utils import save_data

class DataIngestionConfig:
      def __init__(self):
            self.train_data_path = os.path.join('artifacts', 'train.csv')
            self.test_data_path = os.path.join('artifacts', 'test.csv')
            self.raw_data_path = os.path.join('artifacts', 'data.csv')
class DataIngestion:
      def __init__(self):
            self.data_ingestion = DataIngestionConfig()

      def initiate_data_ingestion(self):
            try:
                  logging.info("Data Initiate start.")
                  df = pd.read_csv(r"data_sets\Iris.csv")

                  save_data(df, self.data_ingestion.raw_data_path)

                  # Split data
                  train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
                  save_data(train_set, self.data_ingestion.train_data_path)
                  save_data(test_set, self.data_ingestion.test_data_path)
                  logging.info("Train and test data saved to artifacts folder.")
                  logging.info("Initial data ingestion completed.")
                  
                  return self.data_ingestion.train_data_path, self.data_ingestion.test_data_path

            except Exception as e:
                  raise CustomError(str(e), sys)
if __name__=='__main__':
      data_ingestion = DataIngestion()
      train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()