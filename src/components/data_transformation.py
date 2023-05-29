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
        
        ## INCOMPLETE _____________________________________
        
            logging.info('Pipelining complete')
        
        except Exception as e:
            logging.info('Exception during DataTransformation')
            raise CustomException(e, sys)        