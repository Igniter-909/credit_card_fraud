import sys
import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass
class Data_Transformation_Config:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class Data_Transformation:
    def __init__(self):
        self.data_transformation_config = Data_Transformation_Config()

    def get_data_transformation_object(self):
            
        try:
            columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
            a_pipeline = Pipeline(
                steps=[
                    ("simple_imputer",SimpleImputer(strategy='mean'))
                ]
            )
            logging.info(f"Columns :{columns}")

            preprocessor = ColumnTransformer(
                [
                    ("a_pipeline",a_pipeline,columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info(f"Initializing data transformation for train and test data")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read train and test data")

            preprocessor = self.get_data_transformation_object()
            logging.info("Obtained the preprocessor object")

            target_column_name = ['Class']

            train_data_features_df = train_data.drop(columns=target_column_name,axis=1)
            train_data_target_df = train_data[target_column_name]

            test_data_features_df = test_data.drop(columns=target_column_name,axis=1)
            test_data_target_df = test_data[target_column_name]

            logging.info("Splitted the data into features and target")

            logging.info("Applying the preprocessing object")

            train_data_features_arr = preprocessor.fit_transform(train_data_features_df)
            test_data_features_arr = preprocessor.transform(test_data_features_df)
            logging.info("Applied the preprocessing object")

            train_arr = np.c_[
                train_data_features_arr,
                np.array(train_data_target_df)
            ]
            test_arr = np.c_[
                test_data_features_arr,
                np.array(test_data_target_df)
            ]

            logging.info("Saved preprocessor object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
    
