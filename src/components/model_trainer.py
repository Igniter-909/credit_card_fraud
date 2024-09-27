import os
import sys
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import SVC

from dataclasses import dataclass
from sklearn.metrics import balanced_accuracy_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Starting model trainer")
            logging.info("Training and test splitting performing")

            X_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            X_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]

            models = {
                "Logistic Regression": LogisticRegression(penalty='l2',C=1.0,solver='lbfgs', max_iter=400,random_state=42),
                "DecisionTree Classifier": DecisionTreeClassifier(criterion='gini',max_depth=None,min_samples_split=2,random_state=42),
                "Random Forest Classifier": RandomForestClassifier(n_estimators=100, criterion='gini',max_depth=None,min_samples_split=2, random_state=42),
                "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42),
                "Xg Boost": XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,subsample=0.8,colsample_bytree=0.8,random_state=42),
                "XGBClassifier": XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,random_state=42),
                "SVM": SVC(C=1.0,kernel='rbf',probability=True,gamma='scale',random_state=42)
            }

            model_report:dict = evaluate_model_report(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both traininig and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            score = balanced_accuracy_score(y_test,predicted)
            return score
        
        except Exception as e:
            raise CustomException(e,sys)
