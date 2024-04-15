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
            "LinearRegression":LinearRegression(),
            "CatBoostClassifier":CatBoostClassifier()
            }
            params={
                "DecisionTreeClassifier": {
                  'criterion': ['gini', 'entropy']
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                   "Gradeient Boosting": {
                   'learning_rate': [.1, .01, .05, .001],
                   'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                   'n_estimators': [8, 16, 32, 64, 128, 256]
                   },
                "Linear Regression":{},
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
        
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
        
        