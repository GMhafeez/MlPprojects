import os
import sys

from dataclasses import dataclass

from catboost import CatBoostClassifier

from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import Custom_exception
from src.logger import logging

from src.utils import save_object, evaluate_models 

@dataclass

class modelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()


    def initaite_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models ={
            "RandomForestRegressor":RandomForestRegressor(),
            "DecisionTreeClassifier":DecisionTreeClassifier(),
            "Gradeient Boosting":GradientBoostingClassifier(),
            "AdaBoost":AdaBoostClassifier(),
            "KNN":KNeighborsClassifier(),
            "LogisticRegression":LogisticRegression(),
            "LinearRegression":LinearRegression()
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,)
        
            logging.info("Train and test data split completed")

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise Custom_exception("no best model found",sys.exc_info())
            logging.info(f"best model found is {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)

            return r2_square
        except Exception as e:
            raise Custom_exception( e, sys.exc_info())
        
        