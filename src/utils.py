import os
import dill
import sys
from src.exception import Custom_exception
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
    