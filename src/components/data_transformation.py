# Feature Engineering and Feature Selection
import gc
import warnings
import numpy as np 
import pandas as pd 
import multiprocessing as mp
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder
from category_encoders.binary import BinaryEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbpipeline 
import matplotlib.pyplot as plt

from src.logger  import logging
from src.exception import CustomException
import os
import sys
import pickle
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
            self.data_transformation_config = DataTransformationConfig()
            
    def get_data_transformation_obj(self):
        
        try:
            logging.info('Data transformation initiated')
            
            
            def clean(df):
    
                df['Zip'].fillna(0, inplace=True)
                df['Amount'] = df['Amount'].apply(lambda value: float(value.split("$")[1]))
                df['Hour'] = df['Time'].apply(lambda value: int(value.split(":")[0]))
                df['Minutes'] = df['Time'].apply(lambda value: int(value.split(":")[1]))
                df.drop(['Time'], axis=1, inplace=True)
                df['Merchant State'].fillna('NA', inplace=True)
                df['Merchant State'] = df['Merchant State'].astype('category')
                df['Errors?'].fillna('None', inplace=True)
                df['Errors?'] = df['Errors?'].astype('category')


                cat_col = ['Merchant State','Use Chip', 'Merchant City','Errors?']
                be = BinaryEncoder()
                enc_df= pd.DataFrame(be.fit_transform(df[cat_col]), dtype= 'int8' )  

                df.drop(cat_col, axis=1, inplace = True)
                df = pd.concat([df,enc_df], axis=1)

                for col in df.columns:
                    df[col] =  df[col].astype(float)
                return pd.DataFrame(df)    
     
    
            preprocessing_pipeline = Pipeline([
                    ('cleaning', FunctionTransformer(clean))
                    ], verbose=True) 

            return preprocessing_pipeline 
        
            logging.info('Pipelining complete')
        
        except Exception as e:
            logging.info('Exception during getting DataTransformation object')
            raise CustomException(e, sys)        
        
        
        
        
    def initiate_data_transformation(self,train_data_path, test_data_path): 
        
        
        try : 
            
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logging.info('Read train and test data')
            logging.info(f'Train dataframe head : \n {train_df.head().to_string()}')
            logging.info(f'Test dataframe head : \n {test_df.head().to_string()}') 
            
            logging.info('Getting preprocessor object')
        
            preprocessor = self.get_data_transformation_obj()
            
            target_column = 'Is Fraud?'
            drop_column = [target_column]
            
            # divinding dataset in dependent and indedpent dataset
            # Training dataset:
            #X_train
            input_feature_train_df = train_df.drop(columns = drop_column, axis = 1 )
            #y_train
            target_feature_train_df =  train_df[target_column]
            
            # Test dataset:
            # X_test
            input_feature_test_df = test_df.drop(columns = drop_column, axis = 1 )
            # y_test
            target_feature_test_df =  test_df[target_column]
        
            # Data Transformation:
            
            input_feature_train_df, target_feature_train_df = RandomUnderSampler(random_state=1613 ,
                                                                                  sampling_strategy= 0.01).fit_resample(input_feature_train_df, target_feature_train_df)

            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            # Encoding 
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)
           
            logging.info('Applying preprocessor on train and test data completed')
            
            # Final df
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] 
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            ## PICKLING THE PREPROCESSOR OBJECT 
            ## PICKLING THE PREPROCESSOR OBJECT 
            ## PICKLING THE PREPROCESSOR OBJECT 
            #------------------------------------------
            
        except Exception as e:
            logging.info('Exception during DataTransformation')   
            raise CustomException(e, sys)    