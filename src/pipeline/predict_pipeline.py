import os
import sys
import pandas as pd
from src.exception import CustomError
from src.logger import logging
from src.utils import load_pickle_file

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def predict(self, input_data):
        try:
            # Load model and preprocessor
            logging.info("Loading preprocessor and model for prediction.")
            preprocessor = load_pickle_file(self.preprocessor_path)
            model = load_pickle_file(self.model_path)

            # Apply preprocessing
            logging.info("Applying preprocessing to input data.")
            input_data_transformed = preprocessor.transform(input_data)

            # Make predictions
            logging.info("Making predictions.")
            predictions = model.predict(input_data_transformed)
            return predictions

        except Exception as e:
            raise CustomError(f"Error in prediction: {str(e)}", sys)

if __name__ == "__main__":
    try:
        # Load new test data for prediction
        logging.info("Loading test data for prediction.")
        test_data_path = os.path.join('data_sets', 'new_test_data.csv')
        test_data = pd.read_csv(test_data_path)

        # Create an instance of PredictPipeline
        predict_pipeline = PredictPipeline()

        # Make predictions
        logging.info("Executing prediction pipeline.")
        predictions = predict_pipeline.predict(test_data)

        # Save predictions
        prediction_output_path = os.path.join('artifacts', 'predictions.csv')
        pd.DataFrame({'predictions': predictions}).to_csv(prediction_output_path, index=False)
        logging.info(f"Predictions saved successfully at {prediction_output_path}.")

    except Exception as e:
        raise CustomError(str(e), sys)
    
#python -m src.pipeline.predict_pipeline