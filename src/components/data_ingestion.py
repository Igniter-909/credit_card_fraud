import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import Data_Transformation
from src.components.data_transformation import Data_Transformation_Config
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionCongig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionCongig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion")
        try:
            df = pd.read_csv('notebooks/data/creditcard.csv')
            logging.info("Succesfully loaded the data")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            legit = df[df.Class==0].sample(n=492)
            fraud = df[df.Class==1]
            logging.info("Seggregated the data into fraud and legit samples")

            df = pd.concat([fraud,legit],axis=0)
            logging.info("Balanced the dataset using concatenation.")


            logging.info("Initiated the train test split")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=2)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Completed the data ingestion")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
    
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    
    data_transformer = Data_Transformation()
    train_arr,test_arr,_ = data_transformer.initiate_data_transformation(train_data_path,test_data_path)
    
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr)


