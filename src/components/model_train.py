#src/components/model_train.py 

import os
import sys
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.exception import CustomError
from src.logger import logging
from src.utils import save_pickle_file, load_pickle_file
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class ModelTrainConfig:
      def __init__(self):
            self.model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrain:
      def __init__(self):
            self.model_train = ModelTrainConfig()
      
      def initiate_model_train(self, X_train, y_train, X_test, y_test):
            try:
                  logging.info("Model train start.")
                  model = RandomForestRegressor()
                  model.fit(X_train, y_train)

                  train_pred = model.predict(X_train)
                  test_pred = model.predict(X_test)

                  train_score = r2_score(y_train, train_pred)
                  test_score = r2_score(y_test, test_pred)

                  logging.info(f"Train r2 score {train_score : .4f}, Test r2 score {test_score :.4f}")

                  save_pickle_file(self.model_train.model_file_path, model)

                  return model

            except Exception as e:
                  raise CustomError(str(e), sys)
if __name__=='__main__':
      train_data = pd.read_csv(os.path.join('artifacts', 'train.csv'))
      test_data = pd.read_csv(os.path.join('artifacts', 'test.csv'))

      X_train = train_data.drop(columns=['Species'])
      y_train = train_data['Species']
      X_test = test_data.drop(columns=['Species'])
      y_test = test_data['Species']

      preprcessor = load_pickle_file(os.path.join('artifacts', 'preprocessor.pkl'))

      # Correct transformation of data
      X_train_transformed = preprcessor.fit_transform(X_train)  # Fit and transform on training data
      X_test_transformed = preprcessor.transform(X_test)        # Only transform on test data


      # Encode the target labels
      label_encoder = LabelEncoder()
      y_train_encoded = label_encoder.fit_transform(y_train)
      y_test_encoded = label_encoder.transform(y_test)

      model_train = ModelTrain()
      model_train.initiate_model_train(X_train_transformed, y_train_encoded, X_test_transformed, y_test_encoded)

