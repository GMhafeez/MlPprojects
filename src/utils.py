import os
import dill
import sys
from src.exception import Custom_exception

from src.logger import logging

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV
import pandas as pd

import numpy as np

def save_object (file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as output:
            dill.dump(obj, output)

    except Exception as e:
        raise Custom_exception(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)  # Fit the model using training data

            y_train_pred = model.predict(X_train)  # Make predictions on training data
            y_test_pred = model.predict(X_test)    # Make predictions on test data

            train_model_score = r2_score(y_train, y_train_pred)  # Evaluate training performance
            test_model_score = r2_score(y_test, y_test_pred)      # Evaluate test performance

            report[model_name] = test_model_score  # Store test performance in the report

        return report

    except Exception as e:
        raise Custom_exception(e, sys.exc_info())
        
