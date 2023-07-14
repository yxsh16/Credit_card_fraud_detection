# Reading data from source
# Train test split
import os 
import sys
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.model_selection import train_test_split



@dataclass 
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts' , 'train_data.csv')
    test_data_path:str = os.path.join('artifacts' , 'test_data.csv')
    raw_data_path:str = os.path.join('artifacts' , 'raw_data.csv')




class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('Data ingestion initiated')
        
        try:
            df = pd.read_csv(os.path.join('notebook/dataset' , 'credit_card_transaction.csv'), dtype = {'Year' : 'int16', 'Month' : 'int8', 'Day' : 'int8',
                           'Use Chip' : 'category', 'MCC' : 'int16', 'Is Fraud?' : 'category' , 
                            'Merchant City' : 'category', 'Amount' : 'string'})
            
            # [NOTE : data read in this function is not the original data due to its huge size ,a sample of the data is used
            # please replace credit_card_transaction.csv file with original dataset and then proceeed with training step]
            
            logging.info('CSV file read successfully')
        
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False)

            logging.info('Raw data saved successfully')
        
            train_set, test_set = train_test_split(df, test_size = 0.3, random_state = 1613, shuffle=True)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False , header = True)
            
            logging.info('Train and test data saved successfully')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
            
            
        except Exception as e:
            logging.info('Exception during Data ingestion')
            raise CustomException(e, sys)
                