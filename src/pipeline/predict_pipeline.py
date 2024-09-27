import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predicts(self,features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            features = features.apply(pd.to_numeric,errors='coerce')
            #data_scaled = preprocessor.transform(features)
            preds = model.predict(features)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                Time:float,
                V1: float,
                V2: float,
                V3: float,
                V4: float,
                V5: float,
                V6: float,
                V7: float,
                V8: float,
                V9: float,
                V10: float,
                V11: float,
                V12: float,
                V13: float,
                V14: float,
                V15: float,
                V16: float,
                V17: float,
                V18: float,
                V19: float,
                V20: float,
                V21: float,
                V22: float,
                V23: float,
                V24: float,
                V25: float,
                V26: float,
                V27: float,
                V28: float,
                Amount:float):
        self.time = Time,
        self.v1 = V1,
        self.v2 = V2,
        self.v3 = V3,
        self.v4 = V4,
        self.v5 = V5,
        self.v6 = V6,
        self.v7 = V7,
        self.v8 = V8,
        self.v9 = V9,
        self.v10 = V10,
        self.v11 = V11,
        self.v12 = V12,
        self.v13 = V13,
        self.v14 = V14,
        self.v15 = V15,
        self.v16 = V16,
        self.v17 = V17,
        self.v18 = V18,
        self.v19 = V19,
        self.v20 = V20,
        self.v21 =V21,
        self.v22 = V22,
        self.v23 = V23,
        self.v24 =V24,
        self.v25 = V25,
        self.v26 = V26,
        self.v27 = V27,
        self.v28 = V28,
        self.amount = Amount
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "Time":[self.time],
                "V1": [self.v1],
                "V2": [self.v2],
                "V3": [self.v3],
                "V4": [self.v4],
                "V5": [self.v5],
                "V6": [self.v6],
                "V7": [self.v7],
                "V8": [self.v8],
                "V9": [self.v9],
                "V10": [self.v10],
                "V11": [self.v11],
                "V12": [self.v12],
                "V13": [self.v13],
                "V14": [self.v14],
                "V15": [self.v15],
                "V16": [self.v16],
                "V17": [self.v17],
                "V18": [self.v18],
                "V19": [self.v19],
                "V20": [self.v20],
                "V21": [self.v21],
                "V22": [self.v22],
                "V23": [self.v23],
                "V24": [self.v24],
                "V25": [self.v25],
                "V26": [self.v26],
                "V27": [self.v27],
                "V28": [self.v28],
                "Amount": [self.amount]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
        
    
