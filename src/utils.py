# Common functionalities like reading and writing files, uploading files, pickling the model 

import os 
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import classification_report



def save_object(file_path, object):
    
    try: 
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(object, file_obj)
            
            
    except Exception as e:
        raise CustomException(e, sys)
    
    
    
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
        
    try:
        report ={}
        for i in range (len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)                
            
            logging.info('Model is being trained')
                
            y_pred = model.predict(X_test)
                
            logging.info('Predicting the values')
                
            test_model_report = (classification_report(y_test, y_pred))
                
            report[list(models.keys())[i]] = test_model_report
                
        return report
        
        
        
    except Exception as e:
        logging.info('Error during prediction')
        raise CustomException(e, sys)        