import sys
import os
from src.exception import Custom_exception
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import dataTransformationConfig
@dataclass
class DataingestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class Dataingestion:
    def __init__(self):
        self.ingestion_config = DataingestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion component")
        try:
            df = pd.read_csv('notebook/stud.csv')            
            logging.info('Reading data successfully')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train-test split initiated')

            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info('Ingestion of data completed successfully')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise Custom_exception(str(e), sys.exc_info())

if __name__ == "__main__":
    obj = Dataingestion()
    train_data,test_data = obj.initiate_data_ingestion()


    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
