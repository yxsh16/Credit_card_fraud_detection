import pandas as pd
from src.utils import load_object
import sys
import os
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        model_path = os.path.join('artifacts', 'model.pkl')

        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)

        
        transformed_features = preprocessor.transform(features)
        logging.info(f'Scaled data for prediction: {transformed_features}')

        pred = model.predict(transformed_features)
        return pred


def apply_prediction_pipeline(input_file, output_file):
    try:
        prediction_pipeline = PredictPipeline()
        df = pd.read_csv(input_file)

        predictions = prediction_pipeline.predict(df)

        df['Is Fraud?'] = predictions

        df.to_csv(output_file, index=False)
        
        logging.info(f"Predictions saved to {output_file}")

    except Exception as e:
        logging.error("Exception occurred in prediction")
        raise CustomException(e, sys)
