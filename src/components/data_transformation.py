import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from category_encoders.binary import BinaryEncoder
from imblearn.under_sampling import RandomUnderSampler

from src.logger import logging
from src.exception import CustomException
import os
import sys
import pickle
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


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

    cat_col = ['Merchant State', 'Use Chip', 'Merchant City', 'Errors?']
    be = BinaryEncoder()
    enc_df = pd.DataFrame(be.fit_transform(df[cat_col]), dtype='int8')

    df.drop(cat_col, axis=1, inplace=True)
    df = pd.concat([df, enc_df], axis=1)

    for col in df.columns:
        df[col] = df[col].astype(float)
    
    print(f'DF HEAD : {df.head()}')
    print(f'DF Cols: {df.columns}')
    return df 


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info('Data transformation initiated')

            preprocessing_pipeline = Pipeline([
                ('cleaning', FunctionTransformer(clean))
            ], verbose=True)

            logging.info('Pipelining complete')
            return preprocessing_pipeline

        except Exception as e:
            logging.error('Exception during getting DataTransformation object')
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Read train and test data')
            logging.info(f'Train dataframe head:\n{train_df.head().to_string()}')
            logging.info(f'Test dataframe head:\n{test_df.head().to_string()}')

            logging.info('Getting preprocessor object')

            preprocessor = self.get_data_transformation_obj()

            target_column = 'Is Fraud?'
            drop_column = [target_column]

            X_train = train_df.drop(columns=drop_column)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=drop_column)
            y_test = test_df[target_column]

            X_train, y_train = RandomUnderSampler(
                random_state=1613, sampling_strategy=0.01
            ).fit_resample(X_train, y_train)

            X_train = pd.DataFrame(preprocessor.fit_transform(X_train), dtype = 'float')
            X_test = pd.DataFrame(preprocessor.transform(X_test), dtype = 'float')

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            train_df = pd.concat([X_train, pd.Series(y_train, name=target_column)], axis=1)
            test_df = pd.concat([X_test, pd.Series(y_test, name=target_column)], axis=1)

            logging.info(f'Train transformed dataframe head:\n{train_df.head().to_string()}')
            logging.info(f'Test transformed dataframe head:\n{test_df.head().to_string()}')

            with open(self.data_transformation_config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            return (
                train_df,
                test_df,
                preprocessor,
            )

        except Exception as e:
            logging.error('Exception during DataTransformation')
            raise CustomException(e, sys)
